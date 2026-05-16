#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <napi.h>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
#include "ds4.h"
}

struct ChatMessage {
  std::string role;
  std::string content;
};

struct GenerateParams {
  std::vector<ChatMessage> messages;
  int max_tokens = 2048;
  float temperature = 1.0f;
  float top_p = 1.0f;
  int top_k = 0;
  float min_p = 0.0f;
  uint64_t seed = 0;
  std::vector<std::string> stop_sequences;
};

struct GenerateResult {
  std::string text;
  int prompt_tokens = 0;
  int completion_tokens = 0;
  std::string finish_reason = "stop";
  std::string error_message;
};

struct ModelState {
  ds4_engine *engine = nullptr;
  ds4_session *session = nullptr;
  int ctx_size = 32768;
  std::mutex mutex;

  ~ModelState() {
    if (session != nullptr) {
      ds4_session_free(session);
    }
    if (engine != nullptr) {
      ds4_engine_close(engine);
    }
  }
};

class GenerateWorker;
class StreamGenerateWorker;

static std::unordered_map<int, std::unique_ptr<ModelState>> g_models;
static std::mutex g_models_mutex;
static std::atomic<int> g_next_handle{1};
static std::unordered_map<int, std::vector<GenerateWorker *>> g_generate_workers;
static std::unordered_map<int, std::vector<StreamGenerateWorker *>> g_stream_workers;
static std::mutex g_workers_mutex;

template <typename Worker>
void RemoveWorker(std::unordered_map<int, std::vector<Worker *>> &workers_by_handle, int handle,
                  Worker *worker) {
  auto it = workers_by_handle.find(handle);
  if (it == workers_by_handle.end()) {
    return;
  }

  auto &workers = it->second;
  workers.erase(std::remove(workers.begin(), workers.end(), worker), workers.end());
  if (workers.empty()) {
    workers_by_handle.erase(it);
  }
}

static ds4_backend ParseBackend(const std::string &backend) {
  if (backend == "cpu") {
    return DS4_BACKEND_CPU;
  }
  if (backend == "cuda") {
    return DS4_BACKEND_CUDA;
  }
#ifdef __APPLE__
  return DS4_BACKEND_METAL;
#else
  return DS4_BACKEND_CPU;
#endif
}

static uint64_t DefaultSeed() {
  return static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

static bool EndsWithStopSequence(const std::string &text,
                                 const std::vector<std::string> &stop_sequences,
                                 size_t *stop_start) {
  for (const auto &stop : stop_sequences) {
    if (stop.empty() || text.size() < stop.size()) {
      continue;
    }
    if (text.compare(text.size() - stop.size(), stop.size(), stop) == 0) {
      *stop_start = text.size() - stop.size();
      return true;
    }
  }
  return false;
}

static ds4_tokens BuildPrompt(ModelState *model, const std::vector<ChatMessage> &messages) {
  ds4_tokens prompt{};
  ds4_chat_begin(model->engine, &prompt);

  for (const auto &message : messages) {
    ds4_chat_append_message(model->engine, &prompt, message.role.c_str(), message.content.c_str());
  }

  ds4_chat_append_assistant_prefix(model->engine, &prompt, DS4_THINK_NONE);
  return prompt;
}

static GenerateResult RunGeneration(ModelState *model, const GenerateParams &params,
                                    const std::atomic<bool> &cancelled,
                                    const std::function<bool(const std::string &)> &on_token) {
  std::lock_guard<std::mutex> model_lock(model->mutex);

  GenerateResult result;
  char err[256] = {0};
  ds4_tokens prompt = BuildPrompt(model, params.messages);
  result.prompt_tokens = prompt.len;

  if (ds4_session_sync(model->session, &prompt, err, sizeof(err)) != 0) {
    result.finish_reason = "error";
    result.error_message = err;
    ds4_tokens_free(&prompt);
    return result;
  }

  int max_tokens = params.max_tokens;
  const int room = ds4_session_ctx(model->session) - ds4_session_pos(model->session);
  if (room <= 1) {
    max_tokens = 0;
  } else if (max_tokens > room - 1) {
    max_tokens = room - 1;
  }

  uint64_t rng = params.seed != 0 ? params.seed : DefaultSeed();
  for (int generated = 0; generated < max_tokens && !cancelled.load(); generated++) {
    const int token = ds4_session_sample(model->session, params.temperature, params.top_k,
                                         params.top_p, params.min_p, &rng);
    if (token == ds4_token_eos(model->engine)) {
      result.finish_reason = "stop";
      break;
    }

    if (ds4_session_eval(model->session, token, err, sizeof(err)) != 0) {
      result.finish_reason = "error";
      result.error_message = err;
      break;
    }

    size_t piece_len = 0;
    char *piece = ds4_token_text(model->engine, token, &piece_len);
    if (piece != nullptr && piece_len > 0) {
      result.text.append(piece, piece_len);
      if (on_token && !on_token(std::string(piece, piece_len))) {
        free(piece);
        break;
      }
    }
    free(piece);

    result.completion_tokens++;
    size_t stop_start = 0;
    if (EndsWithStopSequence(result.text, params.stop_sequences, &stop_start)) {
      result.text.resize(stop_start);
      result.finish_reason = "stop";
      break;
    }

    if (generated + 1 == max_tokens) {
      result.finish_reason = "length";
    }
  }

  ds4_tokens_free(&prompt);
  return result;
}

static Napi::Object ToJSResult(Napi::Env env, const GenerateResult &result) {
  Napi::Object object = Napi::Object::New(env);
  object.Set("text", Napi::String::New(env, result.text));
  object.Set("promptTokens", Napi::Number::New(env, result.prompt_tokens));
  object.Set("completionTokens", Napi::Number::New(env, result.completion_tokens));
  object.Set("finishReason", Napi::String::New(env, result.finish_reason));
  if (!result.error_message.empty()) {
    object.Set("errorMessage", Napi::String::New(env, result.error_message));
  }
  return object;
}

static std::vector<ChatMessage> ParseMessages(Napi::Array messages) {
  std::vector<ChatMessage> result;
  for (uint32_t i = 0; i < messages.Length(); i++) {
    Napi::Object message = messages.Get(i).As<Napi::Object>();
    result.push_back({
        message.Get("role").As<Napi::String>().Utf8Value(),
        message.Get("content").As<Napi::String>().Utf8Value(),
    });
  }
  return result;
}

static GenerateParams ParseGenerateParams(Napi::Object options) {
  GenerateParams params;
  params.messages = ParseMessages(options.Get("messages").As<Napi::Array>());
  params.max_tokens =
      options.Has("maxTokens") ? options.Get("maxTokens").As<Napi::Number>().Int32Value() : 2048;
  params.temperature =
      options.Has("temperature") ? options.Get("temperature").As<Napi::Number>().FloatValue() : 1.0f;
  params.top_p = options.Has("topP") ? options.Get("topP").As<Napi::Number>().FloatValue() : 1.0f;
  params.top_k = options.Has("topK") ? options.Get("topK").As<Napi::Number>().Int32Value() : 0;
  params.min_p = options.Has("minP") ? options.Get("minP").As<Napi::Number>().FloatValue() : 0.0f;
  params.seed = options.Has("seed")
                    ? static_cast<uint64_t>(options.Get("seed").As<Napi::Number>().Int64Value())
                    : 0;

  if (options.Has("stopSequences") && options.Get("stopSequences").IsArray()) {
    Napi::Array stops = options.Get("stopSequences").As<Napi::Array>();
    for (uint32_t i = 0; i < stops.Length(); i++) {
      params.stop_sequences.push_back(stops.Get(i).As<Napi::String>().Utf8Value());
    }
  }

  return params;
}

static ModelState *GetModel(int handle) {
  std::lock_guard<std::mutex> lock(g_models_mutex);
  auto it = g_models.find(handle);
  return it == g_models.end() ? nullptr : it->second.get();
}

class LoadModelWorker : public Napi::AsyncWorker {
public:
  LoadModelWorker(Napi::Function &callback, Napi::Object options)
      : Napi::AsyncWorker(callback),
        model_path_(options.Get("modelPath").As<Napi::String>().Utf8Value()),
        mtp_path_(options.Has("mtpPath") ? options.Get("mtpPath").As<Napi::String>().Utf8Value()
                                         : ""),
        backend_(options.Has("backend") ? options.Get("backend").As<Napi::String>().Utf8Value()
                                        : ""),
        ctx_size_(options.Has("contextSize")
                      ? options.Get("contextSize").As<Napi::Number>().Int32Value()
                      : 32768),
        threads_(options.Has("threads") ? options.Get("threads").As<Napi::Number>().Int32Value()
                                        : 0),
        mtp_draft_tokens_(options.Has("mtpDraftTokens")
                              ? options.Get("mtpDraftTokens").As<Napi::Number>().Int32Value()
                              : 0),
        mtp_margin_(options.Has("mtpMargin") ? options.Get("mtpMargin").As<Napi::Number>().FloatValue()
                                             : 0.0f),
        warm_weights_(options.Has("warmWeights") &&
                      options.Get("warmWeights").As<Napi::Boolean>().Value()),
        quality_(options.Has("quality") && options.Get("quality").As<Napi::Boolean>().Value()) {}

  void Execute() override {
    auto model = std::make_unique<ModelState>();
    model->ctx_size = ctx_size_;

    ds4_engine_options options{};
    options.model_path = model_path_.c_str();
    options.mtp_path = mtp_path_.empty() ? nullptr : mtp_path_.c_str();
    options.backend = ParseBackend(backend_);
    options.n_threads = threads_;
    options.mtp_draft_tokens = mtp_draft_tokens_;
    options.mtp_margin = mtp_margin_;
    options.warm_weights = warm_weights_;
    options.quality = quality_;

    if (ds4_engine_open(&model->engine, &options) != 0) {
      SetError("Failed to open DS4 model: " + model_path_);
      return;
    }
    if (ds4_session_create(&model->session, model->engine, ctx_size_) != 0) {
      SetError("Failed to create DS4 session");
      return;
    }

    handle_ = g_next_handle++;
    {
      std::lock_guard<std::mutex> lock(g_models_mutex);
      g_models[handle_] = std::move(model);
    }
  }

  void OnOK() override {
    Callback().Call({Env().Null(), Napi::Number::New(Env(), handle_)});
  }

  void OnError(const Napi::Error &error) override {
    Callback().Call({Napi::String::New(Env(), error.Message()), Env().Null()});
  }

private:
  std::string model_path_;
  std::string mtp_path_;
  std::string backend_;
  int ctx_size_;
  int threads_;
  int mtp_draft_tokens_;
  float mtp_margin_;
  bool warm_weights_;
  bool quality_;
  int handle_ = -1;
};

class GenerateWorker : public Napi::AsyncWorker {
public:
  GenerateWorker(Napi::Function &callback, int handle, GenerateParams params)
      : Napi::AsyncWorker(callback), handle_(handle), params_(std::move(params)) {}

  void Cancel() { cancelled_.store(true); }

  void Execute() override {
    ModelState *model = GetModel(handle_);
    if (model == nullptr) {
      SetError("Invalid DS4 model handle");
      return;
    }

    result_ = RunGeneration(model, params_, cancelled_, nullptr);
  }

  void OnOK() override {
    {
      std::lock_guard<std::mutex> lock(g_workers_mutex);
      RemoveWorker(g_generate_workers, handle_, this);
    }
    Callback().Call({Env().Null(), ToJSResult(Env(), result_)});
  }

  void OnError(const Napi::Error &error) override {
    {
      std::lock_guard<std::mutex> lock(g_workers_mutex);
      RemoveWorker(g_generate_workers, handle_, this);
    }
    Callback().Call({Napi::String::New(Env(), error.Message()), Env().Null()});
  }

private:
  int handle_;
  GenerateParams params_;
  GenerateResult result_;
  std::atomic<bool> cancelled_{false};
};

class StreamGenerateWorker : public Napi::AsyncWorker {
public:
  StreamGenerateWorker(Napi::Function &callback, int handle, GenerateParams params,
                       Napi::ThreadSafeFunction tsfn)
      : Napi::AsyncWorker(callback), handle_(handle), params_(std::move(params)), tsfn_(tsfn) {}

  void Cancel() { cancelled_.store(true); }

  void Execute() override {
    ModelState *model = GetModel(handle_);
    if (model == nullptr) {
      SetError("Invalid DS4 model handle");
      return;
    }

    result_ = RunGeneration(model, params_, cancelled_, [this](const std::string &token) {
      if (cancelled_.load()) {
        return false;
      }

      auto *copy = new std::string(token);
      napi_status status = tsfn_.BlockingCall(
          copy, [](Napi::Env env, Napi::Function callback, std::string *data) {
            callback.Call({Napi::String::New(env, *data)});
            delete data;
          });
      return status == napi_ok && !cancelled_.load();
    });
  }

  void OnOK() override {
    {
      std::lock_guard<std::mutex> lock(g_workers_mutex);
      RemoveWorker(g_stream_workers, handle_, this);
    }
    tsfn_.Release();
    Callback().Call({Env().Null(), ToJSResult(Env(), result_)});
  }

  void OnError(const Napi::Error &error) override {
    {
      std::lock_guard<std::mutex> lock(g_workers_mutex);
      RemoveWorker(g_stream_workers, handle_, this);
    }
    tsfn_.Release();
    Callback().Call({Napi::String::New(Env(), error.Message()), Env().Null()});
  }

private:
  int handle_;
  GenerateParams params_;
  GenerateResult result_;
  Napi::ThreadSafeFunction tsfn_;
  std::atomic<bool> cancelled_{false};
};

Napi::Value LoadModel(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsFunction()) {
    Napi::TypeError::New(env, "Expected (options, callback)").ThrowAsJavaScriptException();
    return env.Null();
  }

  auto callback = info[1].As<Napi::Function>();
  auto worker = new LoadModelWorker(callback, info[0].As<Napi::Object>());
  worker->Queue();
  return env.Undefined();
}

Napi::Value UnloadModel(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expected model handle").ThrowAsJavaScriptException();
    return env.Null();
  }

  const int handle = info[0].As<Napi::Number>().Int32Value();
  {
    std::lock_guard<std::mutex> lock(g_models_mutex);
    g_models.erase(handle);
  }
  return Napi::Boolean::New(env, true);
}

Napi::Value Generate(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3 || !info[0].IsNumber() || !info[1].IsObject() || !info[2].IsFunction()) {
    Napi::TypeError::New(env, "Expected (handle, options, callback)").ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Object options = info[1].As<Napi::Object>();
  if (!options.Has("messages") || !options.Get("messages").IsArray()) {
    Napi::TypeError::New(env, "Expected messages array in options").ThrowAsJavaScriptException();
    return env.Null();
  }

  const int handle = info[0].As<Napi::Number>().Int32Value();
  auto callback = info[2].As<Napi::Function>();
  auto worker = new GenerateWorker(callback, handle, ParseGenerateParams(options));
  {
    std::lock_guard<std::mutex> lock(g_workers_mutex);
    g_generate_workers[handle].push_back(worker);
  }
  worker->Queue();
  return env.Undefined();
}

Napi::Value GenerateStream(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 4 || !info[0].IsNumber() || !info[1].IsObject() || !info[2].IsFunction() ||
      !info[3].IsFunction()) {
    Napi::TypeError::New(env, "Expected (handle, options, tokenCallback, doneCallback)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Object options = info[1].As<Napi::Object>();
  if (!options.Has("messages") || !options.Get("messages").IsArray()) {
    Napi::TypeError::New(env, "Expected messages array in options").ThrowAsJavaScriptException();
    return env.Null();
  }

  const int handle = info[0].As<Napi::Number>().Int32Value();
  auto token_callback = info[2].As<Napi::Function>();
  auto done_callback = info[3].As<Napi::Function>();
  Napi::ThreadSafeFunction tsfn =
      Napi::ThreadSafeFunction::New(env, token_callback, "DS4TokenCallback", 0, 1);

  auto worker = new StreamGenerateWorker(done_callback, handle, ParseGenerateParams(options), tsfn);
  {
    std::lock_guard<std::mutex> lock(g_workers_mutex);
    g_stream_workers[handle].push_back(worker);
  }
  worker->Queue();
  return env.Undefined();
}

Napi::Value CancelGeneration(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expected model handle").ThrowAsJavaScriptException();
    return env.Null();
  }

  const int handle = info[0].As<Napi::Number>().Int32Value();
  bool cancelled = false;
  {
    std::lock_guard<std::mutex> lock(g_workers_mutex);
    auto generate_it = g_generate_workers.find(handle);
    if (generate_it != g_generate_workers.end()) {
      for (GenerateWorker *worker : generate_it->second) {
        worker->Cancel();
        cancelled = true;
      }
    }
    auto stream_it = g_stream_workers.find(handle);
    if (stream_it != g_stream_workers.end()) {
      for (StreamGenerateWorker *worker : stream_it->second) {
        worker->Cancel();
        cancelled = true;
      }
    }
  }
  return Napi::Boolean::New(env, cancelled);
}

Napi::Value IsModelLoaded(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expected model handle").ThrowAsJavaScriptException();
    return env.Null();
  }

  const int handle = info[0].As<Napi::Number>().Int32Value();
  return Napi::Boolean::New(env, GetModel(handle) != nullptr);
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("loadModel", Napi::Function::New(env, LoadModel));
  exports.Set("unloadModel", Napi::Function::New(env, UnloadModel));
  exports.Set("generate", Napi::Function::New(env, Generate));
  exports.Set("generateStream", Napi::Function::New(env, GenerateStream));
  exports.Set("cancelGeneration", Napi::Function::New(env, CancelGeneration));
  exports.Set("isModelLoaded", Napi::Function::New(env, IsModelLoaded));
  return exports;
}

NODE_API_MODULE(ds4_binding, Init)
