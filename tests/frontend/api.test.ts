import assert from "node:assert/strict";
import test from "node:test";
import { DEEPSEEK_IMAGE_UNSUPPORTED_MESSAGE, assertImageRecognitionSupported, isImageRecognitionUnsupported, parseApiResponse } from "../../src/api.ts";

test("non-json API failures explain that the local backend may be offline", async () => {
  const response = new Response("", {
    status: 500,
    headers: { "Content-Type": "text/plain" },
  });

  await assert.rejects(
    () => parseApiResponse(response),
    /本地后端可能没有启动/
  );
});

test("DeepSeek without vision support is blocked, vision models on a gateway are not", () => {
  const deepseekConfig = {
    provider_name: "DeepSeek",
    base_url: "https://api.deepseek.com",
    api_key: "TEST_ONLY_API_KEY",
    model: "deepseek-v4-pro",
    temperature: 0.4,
    max_tokens: 1800
  };
  const openaiConfig = {
    ...deepseekConfig,
    provider_name: "OpenAI",
    base_url: "https://api.openai.com/v1",
    model: "gpt-5.5"
  };
  const gatewayVisionConfig = {
    ...deepseekConfig,
    provider_name: "OpenCode Go",
    base_url: "https://opencode.ai/zen/v1",
    model: "deepseek-v4-flash-vision-exp"
  };
  const gatewayTextConfig = {
    ...gatewayVisionConfig,
    model: "deepseek-v4.1-flash"
  };
  const mislabelledProviderConfig = {
    ...openaiConfig,
    provider_name: "DeepSeek 代理"
  };

  assert.equal(isImageRecognitionUnsupported(deepseekConfig), true);
  assert.equal(isImageRecognitionUnsupported(gatewayTextConfig), true);
  assert.equal(isImageRecognitionUnsupported(gatewayVisionConfig), false);
  assert.equal(isImageRecognitionUnsupported(openaiConfig), false);
  assert.equal(isImageRecognitionUnsupported(mislabelledProviderConfig), false);
  assert.throws(
    () => assertImageRecognitionSupported({ api_config: deepseekConfig }),
    new RegExp(DEEPSEEK_IMAGE_UNSUPPORTED_MESSAGE)
  );
  assert.throws(
    () => assertImageRecognitionSupported({ api_config: gatewayTextConfig }),
    new RegExp(DEEPSEEK_IMAGE_UNSUPPORTED_MESSAGE)
  );
  assert.doesNotThrow(() => assertImageRecognitionSupported({ api_config: openaiConfig }));
  assert.doesNotThrow(() => assertImageRecognitionSupported({ api_config: gatewayVisionConfig }));
  assert.doesNotThrow(() => assertImageRecognitionSupported({ inviteCode: "TEST_ONLY_INVITE" }));
});
