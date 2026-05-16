import {
  DS4LanguageModel,
  type DS4LanguageModelConfig,
  type DS4ProviderSettings,
} from "./ds4-language-model.js";

export interface DS4Provider {
  (settings: DS4ProviderSettings): DS4LanguageModel;
  languageModel(settings: DS4ProviderSettings): DS4LanguageModel;
}

function createDS4(): DS4Provider {
  const provider = (settings: DS4ProviderSettings): DS4LanguageModel => {
    const config: DS4LanguageModelConfig = {
      ...settings,
      modelId: settings.modelId ?? "deepseek-v4-flash",
    };

    return new DS4LanguageModel(config);
  };

  provider.languageModel = provider;

  return provider as DS4Provider;
}

/**
 * Creates a DS4 language model provider.
 *
 * The model runs in-process through a native Node addon linked against DS4.
 */
export const ds4 = createDS4();

export default ds4;
