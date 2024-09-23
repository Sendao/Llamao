const { DEFAULT_PROMPT_CONTEXT } = require("./config");
const { ChatSession } = require("./chat-session");
const { prepareMessagesForIngest } = require("./util");

class InferenceModel {
    llm;
    modelName;
    config;
    activeChatSession;

    constructor(llmodel, config) {
        this.llm = llmodel;
        this.config = config;
        this.modelName = this.llm.name();
    }

    async createChatSession(options) {
        const chatSession = new ChatSession(this, options);
        await chatSession.initialize();
        this.activeChatSession = chatSession;
        return this.activeChatSession;
    }

    async stateSize() {
        return await this.llm.stateSize();
    }
    async reconstructContext() {
        return await this.llm.recalculateContext();
    }
    async recalculateContext() {
        return await this.llm.recalculateContext();
    }
    async selectContext(ctx) {
        return await this.llm.selectContext(ctx);
    }
    async viewContext(ctx) {
        return await this.llm.viewContext(ctx);
    }
    async releaseContext(ctx) {
        return await this.llm.releaseContext(ctx);
    }
    async restoreState(state_data) {
        let buf = Buffer.from(state_data, 'binary');
        return await this.llm.restoreState(buf);
    }
    async tokenLookup(token) {
        return await this.llm.tokenLookup(token);
    }
    async saveState() {
        let sz = await this.stateSize();
        let buf = Buffer.alloc(sz);

        let rv = await this.llm.saveState(buf);
        let tgt = Buffer.alloc(rv);
        buf.copy(tgt, 0, 0, rv);

        return tgt.toString('binary');
    }


    async generate(input, options = DEFAULT_PROMPT_CONTEXT) {
        const { verbose, ...otherOptions } = options;
        const promptContext = {
            promptTemplate: this.config.promptTemplate,
            temp:
                otherOptions.temp ??
                otherOptions.temperature ??
                DEFAULT_PROMPT_CONTEXT.temp,
            ...otherOptions,
        };
        
        if (promptContext.nPast < 0) {
            throw new Error("nPast must be a non-negative integer.");
        }

        if (verbose) {
            console.debug("Generating completion", {
                input,
                promptContext,
            });
        }

        let prompt = input;
        let nPast = promptContext.nPast;
        let tokensIngested = 0;
        let tokensGenerated = 0;
        //console.log(promptContext);
        const result = await this.llm.infer(prompt, {
            ...promptContext,
            nPast,
            onPromptToken: (tokenId, logits, embds) => {
                let continueIngestion = true;
                tokensIngested++;
                if (options.onPromptToken) {
                    // catch errors because if they go through cpp they will loose stacktraces
                    try {
                        // don't cancel ingestion unless user explicitly returns false
                        continueIngestion =
                            options.onPromptToken(tokenId, logits, embds) !== false;
                    } catch (e) {
                        console.error("Error in onPromptToken callback", e);
                        continueIngestion = false;
                    }
                }
                return continueIngestion;
            },
            onResponseToken: (tokenId, token, logits, embds) => {
                let continueGeneration = true;
                tokensGenerated++;
                if (options.onResponseToken) {
                    try {
                        // don't cancel the generation unless user explicitly returns false
                        continueGeneration =
                            options.onResponseToken(tokenId, token, logits, embds) !== false;
                    } catch (err) {
                        console.error("Error in onResponseToken callback", err);
                        continueGeneration = false;
                    }
                }
                return continueGeneration;
            },
        });

        result.tokensGenerated = tokensGenerated;
        result.tokensIngested = tokensIngested;

        if (verbose) {
            console.debug("Finished completion:\n", result);
        }

        return result;
    }

    dispose() {
        this.llm.dispose();
    }
}

class EmbeddingModel {
    llm;
    config;
    MIN_DIMENSIONALITY = 64;
    constructor(llmodel, config) {
        this.llm = llmodel;
        this.config = config;
    }

    embed(text, prefix, dimensionality, do_mean, atlas) {
        return this.llm.embed(text, prefix, dimensionality, do_mean, atlas);
    }

    dispose() {
        this.llm.dispose();
    }
}

module.exports = {
    InferenceModel,
    EmbeddingModel,
};
