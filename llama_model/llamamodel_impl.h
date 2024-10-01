#ifndef LLAMAMODEL_H_I_KNOW_WHAT_I_AM_DOING_WHEN_INCLUDING_THIS_FILE
#error This file is NOT meant to be included outside of llamamodel.cpp. Doing so is DANGEROUS. Be sure to know what you are doing before proceeding to #define LLAMAMODEL_H_I_KNOW_WHAT_I_AM_DOING_WHEN_INCLUDING_THIS_FILE
#endif
#ifndef LLAMAMODEL_H
#define LLAMAMODEL_H

#include <string>
#include <vector>
#include "llmodel.h"

struct LLamaPrivate;
struct EmbModelSpec;

class LLamaModel : public LLModel {
public:
    LLamaModel();
    ~LLamaModel();

    bool supportsEmbedding() const override { return m_supportsEmbedding; }
    bool supportsCompletion() const override { return m_supportsCompletion; }
    bool loadModel(const std::string &modelPath, int n_ctx, int ngl) override;
    bool isModelBlacklisted(const std::string &modelPath) const override;
    bool isEmbeddingModel(const std::string &modelPath) const override;
    bool isModelLoaded() const override;
    size_t requiredMem(const std::string &modelPath, int n_ctx, int ngl) override;
    size_t stateSize() const override;
    size_t saveState(uint8_t *dest) const override;
    size_t restoreState(const uint8_t *src) override;
    bool selectContext(uint8_t ctx_n) override;
    uint8_t viewContext(uint8_t ctx_n) override;
    void releaseContext(uint8_t ctx_n) override;
    void saveImpression() override;
    void pickActor( std::string actorname ) override;
    //void stampMemory() override; // uses pickActor's id
    void setThreadCount(int32_t n_threads) override;
    void markRewind(void) override;
    void rewindToMark(void) override;
    void markGeneration(std::string) override;
    void rewindGeneration(std::string, std::vector<int> &) override;
    void queryActorNames(std::vector<std::string> &) override;
    int32_t threadCount() const override;
    std::vector<GPUDevice> availableGPUDevices(size_t memoryRequired) const override;
    bool initializeGPUDevice(size_t memoryRequired, const std::string &name) const override;
    bool initializeGPUDevice(int device, std::string *unavail_reason = nullptr) const override;
    bool hasGPUDevice() override;
    bool usingGPUDevice() override;
    int reserveCache( PromptContext &ctx, int tokens ) override;
    int pollVocab( std::unordered_map< std::string, int > &searchspace, float *logits ) override;

    size_t embeddingSize() const override;
    // user-specified prefix
    void embed(const std::vector<std::string> &texts, float *embeddings, std::optional<std::string> prefix,
               int dimensionality = -1, size_t *tokenCount = nullptr, bool doMean = true, bool atlas = false,
               EmbedCancelCallback *cancelCb = nullptr) override;
    // automatic prefix
    void embed(const std::vector<std::string> &texts, float *embeddings, bool isRetrieval, int dimensionality = -1,
               size_t *tokenCount = nullptr, bool doMean = true, bool atlas = false) override;

    
    void setKey( std::string keyfor, std::string key, std::string keyval ) override;
    void printTimings( void ) override;
    void unloadActor( std::string actor ) override;
    int evalTokens(std::string inputStr, std::vector<int32_t> &tokens, std::string fromname, std::string toname ) const override;
    void recordMemory(std::string actor, std::string who, std::string when, std::string what ) override;
    void saveActors(void) override;
    void flagTokens(int token0, int token1, int saveflag) const override;
    void toggleFull(int value) const override;
    const char *llamaIdle(PromptContext &ctx, const char **keyptr, const char **fmtptr, int *max_gen) const override;
    std::string tokenLookup(int n) const override;
    void feedData( std::vector<float> &logits, std::vector<float> &embd ) const override;

private:
    LLamaPrivate *d_ptr;
    int myctx;
    bool m_supportsEmbedding = false;
    bool m_supportsCompletion = false;

protected:
    std::vector<Token> tokenize(PromptContext &ctx, const std::string &str, bool special) const override;
    std::string tokenToString(Token id) const override;
    Token sampleToken(PromptContext &ctx, int n_last_batch) const override;
    int32_t contextLength() const override;
    const std::vector<Token> &endTokens() const override;
    bool shouldAddBOS() const override;
    int32_t maxContextLength(std::string const &modelPath) const override;
    int32_t layerCount(std::string const &modelPath) const override;

    void embedInternal(const std::vector<std::string> &texts, float *embeddings, std::string prefix, int dimensionality,
                       size_t *tokenCount, bool doMean, bool atlas, EmbedCancelCallback *cancelCb,
                       const EmbModelSpec *spec);
};

#endif // LLAMAMODEL_H
