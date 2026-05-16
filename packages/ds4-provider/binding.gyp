{
  "targets": [
    {
      "target_name": "ds4_binding",
      "sources": [
        "native/binding.cpp",
        "ds4/ds4.c"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "ds4"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "defines": [
        "NAPI_DISABLE_CPP_EXCEPTIONS"
      ],
      "cflags": [
        "-O3",
        "-ffast-math",
        "-Wall",
        "-Wextra"
      ],
      "cflags_cc": [
        "-std=c++17",
        "-O3",
        "-Wall",
        "-Wextra"
      ],
      "conditions": [
        [
          "OS=='mac'",
          {
            "sources": [
              "ds4/ds4_metal.m"
            ],
            "xcode_settings": {
              "CLANG_ENABLE_OBJC_ARC": "YES",
              "OTHER_CFLAGS": [
                "-O3",
                "-ffast-math",
                "-mcpu=native"
              ],
              "OTHER_CPLUSPLUSFLAGS": [
                "-std=c++17"
              ]
            },
            "libraries": [
              "-framework Foundation",
              "-framework Metal"
            ]
          }
        ],
        [
          "OS!='mac'",
          {
            "defines": [
              "DS4_NO_GPU",
              "_GNU_SOURCE"
            ],
            "cflags": [
              "-march=native",
              "-fno-finite-math-only"
            ],
            "libraries": [
              "-lm",
              "-pthread"
            ]
          }
        ]
      ]
    }
  ]
}
