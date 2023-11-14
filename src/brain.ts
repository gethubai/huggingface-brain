import {
  BrainPromptResponse,
  IBrainService,
  IAudioTranscriberBrainService,
  ITextBrainService,
  LocalAudioPrompt,
  TextBrainPrompt,
  IBrainPromptContext,
  BrainSettingsValidationResult,
  IImageGenerationBrainService,
  ImageGenerationBrainPrompt,
  convertToBuffer,
  convertBufferToExpectedImagePromptResultType,
  readAudioFile,
} from '@hubai/brain-sdk';
import {
  AutomaticSpeechRecognitionArgs,
  ConversationalArgs,
  HfInference,
  TextToImageArgs,
} from '@huggingface/inference';

/* Example of setting required by this brain */
export interface ISettings {
  accessToken: string;
  textModel: string;
  voiceTranscriptionModel?: string;
  imageGenerationModel?: string;
}

async function readStreamToBlob(
  readStream: any,
  fileType: string,
): Promise<Blob> {
  const chunks = [];
  for await (const chunk of readStream) {
    chunks.push(chunk);
  }
  return new Blob(chunks, { type: fileType });
}

/* If your brain does not support AudioTranscription just remove the interface implementation */
export default class HuggingFaceBrainService
  implements
    IBrainService,
    ITextBrainService<ISettings>,
    IAudioTranscriberBrainService<ISettings>,
    IImageGenerationBrainService<ISettings>
{
  client?: HfInference;
  currentKey?: string;

  async transcribeAudio(
    prompt: LocalAudioPrompt,
    context: IBrainPromptContext<ISettings>,
  ): Promise<BrainPromptResponse> {
    const validationResult = this.validateSettings(context.settings);

    if (!validationResult.success) {
      return Promise.resolve({
        result: validationResult.getMessage(),
        validationResult,
      });
    }

    const args: AutomaticSpeechRecognitionArgs = {
      data: await readStreamToBlob(readAudioFile(prompt), 'audio/wav'),
      model: context.settings.voiceTranscriptionModel,
    };

    return this.getClient(context.settings)
      .automaticSpeechRecognition(args)
      .then((result) => {
        return {
          result: result.text,
          validationResult,
        };
      });
  }

  async sendTextPrompt(
    prompts: TextBrainPrompt[],
    context: IBrainPromptContext<ISettings>,
  ): Promise<BrainPromptResponse> {
    const validationResult = this.validateSettings(context.settings);

    // If the settings are not valid we return the validation result
    if (!validationResult.success) {
      return Promise.resolve({
        result: validationResult.getMessage(),
        validationResult,
      });
    }

    const prompt = prompts[prompts.length - 1];

    const params: ConversationalArgs = {
      model: context.settings.textModel,
      inputs: {
        past_user_inputs: prompts
          .slice(0, -1)
          .filter((m) => m.role === 'user')
          .map((p) => p.message),
        generated_responses: prompts
          .filter((m) => m.role === 'brain')
          .map((p) => p.message),
        text: prompt.message,
      },
    };

    return this.getClient(context.settings)
      .conversational(params)
      .then((result) => ({ result: result.generated_text, validationResult }));
  }

  generateImage(
    prompts: ImageGenerationBrainPrompt[],
    context: IBrainPromptContext<ISettings>,
  ): Promise<BrainPromptResponse> {
    const validationResult = this.validateSettings(context.settings);

    if (!validationResult.success) {
      return Promise.resolve({
        result: validationResult.getMessage(),
        validationResult,
      });
    }

    const prompt = prompts[prompts.length - 1];
    const args: TextToImageArgs = {
      inputs: prompt.message,
      parameters: {
        negative_prompt: prompt.negativePrompt,
      },
      model: context.settings.imageGenerationModel,
    };

    return this.getClient(context.settings)
      .textToImage(args)
      .then(async (result) => {
        const prompt = prompts[prompts.length - 1];
        const data: Buffer | string =
          convertBufferToExpectedImagePromptResultType(
            await convertToBuffer(result.stream()),
            prompt.expectedResponseType,
          );

        return {
          result: '',
          validationResult,
          attachments: [
            {
              data,
              mimeType: 'image/png',
              fileType: 'image',
            },
          ],
        };
      });
  }

  validateSettings(settings: ISettings): BrainSettingsValidationResult {
    const validation = new BrainSettingsValidationResult();

    const content = `# AccessToken is Missing
    Ops! Looks like you didn't configure your Hugging Face AccessToken. This is required to use this brain.

    ## How to get an AccessToken

    [Log in into your account](https://huggingface.co/login) or [create one](https://huggingface.co/join) if you don't have it.
    After that, go to the [AccessToken page](https://huggingface.co/settings/tokens) and copy your Access Token (or create one if you still don't have).

    ## How to configure your AccessToken at HubAI

    Go to the Brains page, select this brain and set the AccessToken. After that just click on the **Save Settings** button and you're ready to go!
    `;

    /* Example of settings validation */
    if (!settings.accessToken) validation.addError(content);

    return validation;
  }

  getClient(settings: ISettings): HfInference {
    if (!this.client || this.currentKey !== settings.accessToken) {
      this.client = new HfInference(settings.accessToken);
      this.currentKey = settings.accessToken;
    }
    return this.client;
  }
}
