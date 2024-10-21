#include "prompt.h"
#include <future>

PromptWorker::PromptWorker(Napi::Env env, PromptWorkerConfig config)
    : promise(Napi::Promise::Deferred::New(env)), _config(config), AsyncWorker(env)
{
    if (_config.hasResponseCallback)
    {
        _responseCallbackFn = Napi::ThreadSafeFunction::New(config.responseCallback.Env(), config.responseCallback,
                                                            "PromptWorker", 0, 1, this);
    }

    if (_config.hasPromptCallback)
    {
        _promptCallbackFn = Napi::ThreadSafeFunction::New(config.promptCallback.Env(), config.promptCallback,
                                                          "PromptWorker", 0, 1, this);
    }
}

PromptWorker::~PromptWorker()
{
    if (_config.hasResponseCallback)
    {
        _responseCallbackFn.Release();
    }
    if (_config.hasPromptCallback)
    {
        _promptCallbackFn.Release();
    }
}

void PromptWorker::Execute()
{
    _config.mutex->lock();

    LLModelWrapper *wrapper = reinterpret_cast<LLModelWrapper *>(_config.model);

    auto ctx = &_config.context;

    // Copy the C prompt context
    //wrapper->promptContext.n_past = ctx->n_past;
    wrapper->promptContext.n_ctx = ctx->n_ctx;
    wrapper->promptContext.n_predict = ctx->n_predict;
    wrapper->promptContext.top_k = ctx->top_k;
    wrapper->promptContext.top_p = ctx->top_p;
    wrapper->promptContext.temp = ctx->temp;
    wrapper->promptContext.n_batch = ctx->n_batch;
    wrapper->promptContext.repeat_penalty = ctx->repeat_penalty;
    wrapper->promptContext.repeat_last_n = ctx->repeat_last_n;
    wrapper->promptContext.contextErase = ctx->context_erase;
    wrapper->promptContext.continuing = ctx->continuing;

    // Call the C++ prompt method

    wrapper->llModel->prompt(
        _config.prompt, _config.promptTemplate, [this](int32_t token_id, int lsz, int esz, float *logits, float *embds) { return PromptCallback(token_id, lsz, esz, logits, embds); },
        [this](int32_t token_id, const std::string token, int lsz, int esz, float *logits, float *embds) { return ResponseCallback(token_id, token, lsz, esz, logits, embds); },
         wrapper->promptContext, _config.special,
        _config.fakeReply);

    // Update the C context by giving access to the wrappers raw pointers to std::vector data
    // which involves no copies
    ctx->logits = wrapper->promptContext.logits.data();
    ctx->logits_size = wrapper->promptContext.logits.size();
    ctx->tokens = wrapper->promptContext.tokens.data();
    ctx->tokens_size = wrapper->promptContext.tokens.size();

    // Update the rest of the C prompt context
    ctx->n_past = wrapper->promptContext.n_past;
    ctx->n_ctx = wrapper->promptContext.n_ctx;
    ctx->n_predict = wrapper->promptContext.n_predict;
    ctx->top_k = wrapper->promptContext.top_k;
    ctx->top_p = wrapper->promptContext.top_p;
    ctx->temp = wrapper->promptContext.temp;
    ctx->n_batch = wrapper->promptContext.n_batch;
    ctx->repeat_penalty = wrapper->promptContext.repeat_penalty;
    ctx->repeat_last_n = wrapper->promptContext.repeat_last_n;
    ctx->context_erase = wrapper->promptContext.contextErase;
    ctx->continuing = wrapper->promptContext.continuing;

    _config.mutex->unlock();
}

void PromptWorker::OnOK()
{
    Napi::Object returnValue = Napi::Object::New(Env());
    returnValue.Set("text", result);
    returnValue.Set("nPast", _config.context.n_past);
    returnValue.Set("continuing", _config.context.continuing);
    returnValue.Set("nPredict", _config.context.n_predict);
    promise.Resolve(returnValue);
    delete _config.fakeReply;
}

void PromptWorker::OnError(const Napi::Error &e)
{
    delete _config.fakeReply;
    promise.Reject(e.Value());
}

Napi::Promise PromptWorker::GetPromise()
{
    return promise.Promise();
}

bool PromptWorker::ResponseCallback(int32_t token_id, const std::string token, int lsz, int esz, float *logits, float *embds)
{
    if (token_id == -1)
    {
        return false;
    }

    if (!_config.hasResponseCallback)
    {
        return true;
    }

    result += token;

    std::promise<bool> promise;

    auto info = new ResponseCallbackData();
    info->tokenId = token_id;
    info->token = token;
    info->lsz = lsz;
    info->esz = esz;
    info->logits = logits;
    info->embds = embds;

    auto future = promise.get_future();

    auto status = _responseCallbackFn.BlockingCall(
        info, [&promise](Napi::Env env, Napi::Function jsCallback, ResponseCallbackData *value) {
            try
            {
                // Transform native data into JS data, passing it to the provided
                // `jsCallback` -- the TSFN's JavaScript function.
                auto token_id = Napi::Number::New(env, value->tokenId);
                auto token = Napi::String::New(env, value->token);
                //auto mlsz = Napi::Number::New(env, value->lsz);
                //auto mesz = Napi::Number::New(env, value->esz);
                //auto mlogits = Napi::Array::New(env, value->lsz==0?1:value->lsz);
                auto mlogits = Napi::Array::New(env, value->lsz);
                mlogits.Set((uint32_t)0, Napi::Number::New(env, (double)0));
                for( int x = 0; x < value->lsz; x++ ) {
                    mlogits.Set((uint32_t)x, Napi::Number::New(env, value->logits[x]));
                }
                //auto membd = Napi::Array::New(env, value->esz==0?1:value->esz);
                auto membd = Napi::Array::New(env, value->esz);
                membd.Set((uint32_t)0, Napi::Number::New(env, (double)0));
                for( int x = 0; x < value->esz; x++ ) {
                    membd.Set((uint32_t)x, Napi::Number::New(env, value->embds[x]));
                }
                auto jsResult = jsCallback.Call({token_id, token, mlogits, membd}).ToBoolean();
                promise.set_value(jsResult);
            }
            catch (const Napi::Error &e)
            {
                std::cerr << "Error in onResponseToken callback: " << e.what() << std::endl;
                promise.set_value(false);
            }

            delete value;
        });
    if (status != napi_ok)
    {
        Napi::Error::Fatal("PromptWorkerResponseCallback", "Napi::ThreadSafeNapi::Function.NonBlockingCall() failed");
    }

    return future.get();
}

bool PromptWorker::RecalculateCallback(bool isRecalculating)
{
    return isRecalculating;
}

bool PromptWorker::PromptCallback(int32_t token_id, int lsz, int esz, float *logits, float *embds)
{
    if (!_config.hasPromptCallback)
    {
        return true;
    }

    std::promise<bool> promise;

    auto info = new PromptCallbackData();
    info->tokenId = token_id;
    info->lsz = lsz;
    info->esz = esz;
    info->logits = logits;
    info->embds = embds;

    auto future = promise.get_future();

    auto status = _promptCallbackFn.BlockingCall(
        info, [&promise](Napi::Env env, Napi::Function jsCallback, PromptCallbackData *value) {
            try
            {
                // Transform native data into JS data, passing it to the provided
                // `jsCallback` -- the TSFN's JavaScript function.
                auto token_id = Napi::Number::New(env, value->tokenId);
                Napi::Array mlogits = Napi::Array::New(env, value->lsz==0?1:value->lsz);
                mlogits.Set((uint32_t)0, Napi::Number::New(env, (double)0 ) );
                for( int x = 0; x < value->lsz; x++ ) {
                    mlogits.Set((uint32_t)x, Napi::Number::New(env, value->logits[x]) );
                }
                Napi::Array membd = Napi::Array::New(env, value->esz==0?1:value->esz);
                membd.Set((uint32_t)0, Napi::Number::New(env, (double)0 ) );
                for( int x = 0; x < value->esz; x++ ) {
                    membd.Set((uint32_t)x, Napi::Number::New(env, value->embds[x]) );
                }
                auto jsResult = jsCallback.Call({token_id, mlogits, membd}).ToBoolean();
                promise.set_value(jsResult);
            }
            catch (const Napi::Error &e)
            {
                std::cerr << "Error in onPromptToken callback: " << e.what() << std::endl;
                promise.set_value(false);
            }
            delete value;
        });
    if (status != napi_ok)
    {
        Napi::Error::Fatal("PromptWorkerPromptCallback", "Napi::ThreadSafeNapi::Function.NonBlockingCall() failed");
    }

    return future.get();
}
