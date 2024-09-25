#include "llmodel.h"

//#include <cassert>
#include <iostream>
#include <string>

void LLModel::prompt(const std::string &oldprompt,
                     const std::string &promptTemplate,
                     std::function<bool(int32_t, int, int, float *, float *)> promptCallback,
                     std::function<bool(int32_t, const std::string&, int, int, float *, float *)> responseCallback,
                     PromptContext &promptCtx,
                     bool special,
                     std::string *fakeReply)
{
    if (!isModelLoaded()) {
        std::cerr << implementation().modelType() << " ERROR: prompt won't work with an unloaded model!\n";
        return;
    }

    if (!supportsCompletion()) {
        std::string errorMessage = "ERROR: this model does not support text completion or chat!";
        responseCallback(-1, errorMessage, 0, 0, NULL, NULL);
        std::cerr << implementation().modelType() << " " << errorMessage << "\n";
        return;
    }

    if( oldprompt.length() == 0 ) {
        //std::cerr << "Run idle prompt\n";
        //idle_prompt(promptCallback, responseCallback, promptCtx);
        //generateResponse(responseCallback, promptCtx);
        //! use last fromname and generateResponse anyway
        return;
    }

    std::string prompt=oldprompt;

    if( prompt[0] == '&' ) { // for manual loading of old memories
        // loading old memories
        long ipos, npos, tpos, epos;
        uint32_t count;
        std::string actor;
        std::string name, when, value;

        ipos = 0;
        count = 0;
        npos = prompt.find("\n", ipos);
        actor = prompt.substr(1,npos-1);
        ipos = prompt.find("\n", npos+1); // &<actor>\n
        while( ipos != std::string::npos && ipos < prompt.length() ) {
            npos = prompt.find("\n", ipos); // <name>\n<when>:<message>"<store_end>"...
            name = prompt.substr(ipos+1, npos-(ipos+1));
            tpos = prompt.find(":", npos);
            when = prompt.substr(npos+1, tpos-(npos+1));
            ipos = prompt.find("<store_end>", npos);
            value = prompt.substr(tpos+1, ipos-(tpos+1) );
            if( ipos != std::string::npos ) ipos += 10;
            if( count < 10 )
                std::cerr << "recordMemory(" << actor << "," << name << "," << when << "," << value << ")\n";
            recordMemory(actor, name, when, value);
            count++;
        }
        std::cerr << "Finished loading " << count << " memories for " << actor << ".\n";
        return;
    }

    if( prompt[0] == '*' ) { //*key:actor_name\nvalue || // *global_key:\nvalue

        std::string key = "";
        std::string keyfor = "";
        std::string keyval = "";

        int seppos = prompt.find(":");
        int linepos = prompt.find("\n");
        key = prompt.substr(1, seppos-1);
        if( linepos != seppos+1 ) {
            keyfor = prompt.substr(seppos+1, linepos-(seppos+1));
        } else {
            keyfor = "all";
        }
        keyval = prompt.substr(linepos+1);
        setKey(keyfor, key, keyval);

        return;
    }

    std::string toname="all";

    int save_start;

    if( prompt.find("/save") != std::string::npos ) {
        //later we can add saving to a specific place
        saveActors();
        return;
    }

    int pos, parampos;
    std::string param;
    pos = prompt.find("/unload");
    if( pos != std::string::npos ) {
        parampos = pos + 8;
        int ipos = prompt.find("\n", parampos);
        if( ipos == std::string::npos ) {
            ipos = prompt.length();
        }
        param = prompt.substr(parampos, ipos-parampos);
        unloadActor(param);
        return;
    }
    pos = prompt.find("/to");
    if( pos != std::string::npos ) {
        parampos = pos + 4;

        int ipos = prompt.find("\n", parampos);
        if( ipos == std::string::npos ) {
            ipos = prompt.length();
        }
        param = prompt.substr(parampos, ipos-parampos);
        prompt = prompt.substr(0,parampos-4) + prompt.substr(ipos);
        toname = param;
    }
    int usernamepos = prompt.find("<|im_start|>");
    std::string fromname;

    if( usernamepos != std::string::npos ) {
        usernamepos += 12;
        int newlinepos = prompt.find("\n", usernamepos);
        fromname = prompt.substr(usernamepos, newlinepos-usernamepos);
    } else {
        fromname = "user";
        std::cerr << "info: couldn't find fromname\n";
    }
    std::cerr << fromname << " fromname\n";

    if( prompt.compare("_fulldata") == 0 ) {
        toggleFull(2);
        return;
    } else if( prompt.compare("_full") == 0 ) {
        toggleFull(1);
        return;
    } else if( prompt.compare("_short") == 0 ) {
        toggleFull(0);
        return;
    }

    bool norecord = special;
    int bangpos = prompt.find("\n");
    if( bangpos != std::string::npos && bangpos+1 < prompt.length() ) {
        if( prompt[bangpos+1] == '!' ) {
            norecord = true;
        }
    }

    //std::cerr << "LLModel reading prompt " << prompt << "\n";
    // tokenize the user prompt
    //std::vector<Token> embd_inp;
    //embd_inp = tokenize(promptCtx, prompt, special);

    // decode the user prompt
    int n_last_batch = decodePrompt(promptCallback, responseCallback, promptCtx, fromname, toname, prompt);

    std::cerr << "decodePrompt complete\n";

    // decode the assistant's reply, either generated or spoofed
    if (fakeReply == nullptr) {
        generateResponse(responseCallback, promptCtx, fromname, toname, n_last_batch);
    } else {
        //embd_inp = tokenize(promptCtx, *fakeReply, false);
        decodePrompt(promptCallback, responseCallback, promptCtx, fromname, toname, *fakeReply);
    }
}

/*
void LLModel::idle_prompt(std::function<bool(int32_t, int, int, float*, float*)> promptCallback,
                           std::function<bool(int32_t, const std::string&, int, int, float*, float*)> responseCallback,
                          PromptContext &promptCtx)
{
    if( promptCtx.continuing ) {
        generateResponse(responseCallback, promptCtx);
        if( promptCtx.n_predict == 0 ) {
            setKey("");
        }
        return;
    }
    const char *keyptr, *fmtptr;
    int max_gen;
    const char *req = llamaIdle(promptCtx, &keyptr, &fmtptr, &max_gen);
    if( !req ) {
        return;
    }
    std::cerr << __func__ << ": idle function " << req << " " << keyptr << " " << max_gen << " " << fmtptr << "\n";
    std::string *request = new std::string(req);
    if( request->length() == 0 ) return;

    std::string *keycopy = new std::string(keyptr);
    //std::string *fmtcopy = new std::string(fmtptr);

    std::vector<Token> embd_inp = tokenize(promptCtx, *request, true);

    int i, len=embd_inp.size();
    float *zero=NULL;
    std::string x = "idle prompt: ";
    x.append(keycopy->c_str());
    //TODO: Conversion to std::string can be avoided here...
    if (!responseCallback(0, x.c_str(), 0, 0, zero, zero))
        return;
    for( i=0; i<len; i++ ) {
        //TODO: Conversion to std::string can be avoided here...
        if (!responseCallback(embd_inp[i], std::string(tokenToString(embd_inp[i])), 0, 0, zero, zero))
            return;
    }

    decodePrompt(promptCallback, responseCallback, promptCtx, embd_inp);

    promptCtx.n_predict = max_gen;
    promptCtx.continuing = true;
    promptCtx.n_past = reserveCache(promptCtx, max_gen+24); // overrides n_past
    setKey(keycopy->c_str());
    generateResponse(responseCallback, promptCtx);

    if( promptCtx.n_predict == 0 ) {
        setKey("");
    }

    delete(keycopy);
    //delete(fmtcopy);
    delete(request);
}
    */

int LLModel::decodePrompt(std::function<bool(int32_t, int, int, float*, float*)> promptCallback,
                           std::function<bool(int32_t, const std::string&, int, int, float*, float*)> responseCallback,
                           PromptContext &promptCtx,
                           //std::vector<Token> tokens, //embd_inp,
                           std::string fromname,
                           std::string toname,
                           std::string prompt) {
    // save the context size
    promptCtx.n_ctx = contextLength();

    /*
    if ((int) embd_inp.size() > promptCtx.n_ctx - 4) {
        responseCallback(-1, "ERROR: The prompt size exceeds the context window size and cannot be processed.", 0, 0, NULL, NULL);
        std::cerr << implementation().modelType() << " ERROR: The prompt is " << embd_inp.size() <<
            " tokens and the context window is " << promptCtx.n_ctx << "!\n";
        return;
    }
*/


    /*
    promptCtx.n_past = reserveCache(promptCtx, embd_inp.size()+1); // overrides n_past

    if (size_t(promptCtx.n_past) < promptCtx.tokens.size()) {
        std::cerr << "Resize tokens " << promptCtx.tokens.size() << " to " << promptCtx.n_past << "\n";
        promptCtx.tokens.resize(promptCtx.n_past);
    }
    */

    //std::cerr << "LLModel processing prompt (" << embd_inp.size() << ")\n";

    /*
    promptCtx.n_predict = std::min(promptCtx.n_predict, promptCtx.n_ctx - (int) embd_inp.size());
    promptCtx.n_batch = std::min(promptCtx.n_batch, LLMODEL_MAX_PROMPT_BATCH);

    // process the prompt in batches
    size_t i = 0;
    std::string inputStr;
    while (i < embd_inp.size()) {
        size_t batch_end = std::min(i + promptCtx.n_batch, embd_inp.size());
        std::vector<Token> batch(embd_inp.begin() + i, embd_inp.begin() + batch_end);

        inputStr = "";
        for( int j=0; j<batch.size(); j++ ) {
            inputStr.append( tokenToString(batch[j]) );
        }

        if (!evalTokens(inputStr, batch)) {
            std::cerr << implementation().modelType() << " ERROR: Failed to process prompt\n";
            return;
        }

        size_t tokens = batch_end - i;
        for (size_t t = 0; t < tokens; ++t) {
            promptCtx.tokens.push_back(batch.at(t));
            if (!promptCallback(batch.at(t), 0, 0, NULL, NULL))
                return;
        }
        i = batch_end;
        promptCtx.n_past += tokens;
    }
    */
    //std::string empty = "";
    std::vector<int> tokens;
    return evalTokens(prompt, tokens, fromname, toname);
}

void LLModel::generateResponse(std::function<bool(int32_t, const std::string&, int, int, float*, float*)> responseCallback,
                               PromptContext &promptCtx, std::string fromname, std::string toname,
                               int n_last_batch) {
    std::string cachedResponse;
    std::string end_literal = "<|im_end|>";
    std::string new_literal = "<|im_start|>";
    int i;

    // predict next tokens
    int32_t n_gen=promptCtx.n_predict;
    if( n_gen == 0 ) n_gen = 1024;
    if( promptCtx.continuing ) {
        n_gen = n_gen > promptCtx.per_idle ? promptCtx.per_idle : n_gen;
    }

    std::string buf = "";
    std::string sstr;

    std::vector<int> tokenbuf;
    std::vector<std::string> strbuf;
    std::vector<std::vector<float>> logitbuf;
    std::vector<std::vector<float>> embdbuf;

    std::vector<int>::iterator it;
    std::vector<std::string>::iterator sit;
    std::vector<std::vector<float>>::iterator lit, eit;

    int generatedTokens=0;
    bool ending=false;
    bool gen_new=false;
    bool finished_gen=false;
    std::vector<int> newTokens;
    std::string activename="System";
    bool sendToAll=false;

    if( toname == "all" ) {
        sendToAll=true;
    }

    std::cerr << "genResponse(" << n_gen << ": predict " << promptCtx.n_predict << ")\n";
    for (i = 0; i < n_gen; i++) {
        //std::cerr << "tokens.size() = " << promptCtx.tokens.size() << "\n";
        feedData( promptCtx.logits, promptCtx.embds );
        std::cerr << "sampleToken n_last_batch=" << n_last_batch << "\n";
        auto id = sampleToken(promptCtx, n_last_batch);
        newTokens.clear();
        newTokens.push_back(id);
        std::cerr << "tokenToString(" << id << ")\n";
        const std::string str = tokenToString(id);
        std::cerr << "gen: " << str << "(" << id << ")\n";
        if( (n_last_batch=evalTokens(str, newTokens, activename, toname)) == 0 ) {
            std::cerr << implementation().modelType() << " ERROR: Failed to predict next token\n";
            id = 32000; // end
        }
        promptCtx.tokens.emplace_back( id );
        buf += std::string(str);

        auto mlogits = promptCtx.logits;
        auto membd = promptCtx.embds;

        if( gen_new && sendToAll ) {
            int ipos;
            ipos = buf.find(" ");
            if( ipos == std::string::npos ) {
                ipos = buf.find("\n");
            }
            if( ipos == std::string::npos ) {
                ipos = buf.find("\t");
            }
            if( ipos == std::string::npos ) {
                ipos = buf.find(":");
            }
            if( ipos != std::string::npos ) {
                // name completed.
                // switch actor.
                buf = buf.substr(0, ipos);
                std::cerr << "Found actor " << buf << "\n";
                if( buf == fromname ) {
                    // end here so we don't send the tokens back.
                    break;
                }
                activename = buf;
                pickActor(buf);
                gen_new=false;
            }
            tokenbuf.push_back(id);
            strbuf.push_back(std::string(str));
            embdbuf.push_back(membd);
            logitbuf.push_back(mlogits);
            continue;
        }

        if( sendToAll && new_literal.starts_with(buf) ) {
            std::cerr << "found new_literal\n";
            if( buf.starts_with(new_literal) ) {
                std::cerr << "new_literal found buf\n";

                // generate new messenger
                pickActor("System");
                activename = "System";
                finished_gen=false;
                gen_new=true;
                if( buf.length() > new_literal.length() ) {
                    buf = buf.substr(new_literal.length());
                } else {
                    buf = "";
                }

                tokenbuf.push_back(id);
                strbuf.push_back(std::string(str));
                embdbuf.push_back(membd);
                logitbuf.push_back(mlogits);
                continue;
            }
            finished_gen=true;
        }
        if( finished_gen && !ending ) { // we are between generations trying to figure out who speaks next
            tokenbuf.push_back(id);
            strbuf.push_back(std::string(str));
            embdbuf.push_back(membd);
            logitbuf.push_back(mlogits);
            continue;
        }

        if( tokenbuf.size() > 0 ) {
            sit = strbuf.begin();
            lit = logitbuf.begin();
            eit = embdbuf.begin();

            for( it = tokenbuf.begin(); it != tokenbuf.end(); it++ ) {
                id = *it;
                sstr = *sit;
                mlogits = *lit;
                membd = *eit;
                if(!responseCallback(id, sstr, mlogits.size(), membd.size(), mlogits.data(), membd.data())) {
                    promptCtx.n_predict = 0;
                    i = 0;
                    std::cerr << "Generation aborted by responseCallback.\n";
                    ending=true;
                    break;
                }
            }
            tokenbuf.clear();
            logitbuf.clear();
            embdbuf.clear();

            if( ending ) break;
            continue;
        }

        // share with cb
        if (!responseCallback(id, std::string(str), mlogits.size(), membd.size(), mlogits.data(), membd.data())) {
            promptCtx.n_predict = 0;
            i = 0;
            std::cerr << "Generation aborted by responseCallback.\n";
            break;
        }

        if( generatedTokens > 0 && buf.ends_with(end_literal) ) {
            pickActor("System");
            activename="System";
            //promptCtx.n_predict = 0;
            i = 0;
            if( !sendToAll ) { // normal termination
                break;
            } // if sending to all, we let them deliberate by choosing a new speaker.
            // the new speaker may be the user by the way!
            finished_gen = true;
            buf = "";
            generatedTokens++;
            continue;
        }
        if( !end_literal.starts_with(buf) ) {
            generatedTokens++;
            buf="";
        }
    }
}

void LLModel::embed(
    const std::vector<std::string> &texts, float *embeddings, std::optional<std::string> prefix, int dimensionality,
    size_t *tokenCount, bool doMean, bool atlas, EmbedCancelCallback *cancelCb
) {
    (void)texts;
    (void)embeddings;
    (void)prefix;
    (void)dimensionality;
    (void)tokenCount;
    (void)doMean;
    (void)atlas;
    (void)cancelCb;
    throw std::logic_error(std::string(implementation().modelType()) + " does not support embeddings");
}

void LLModel::embed(
    const std::vector<std::string> &texts, float *embeddings, bool isRetrieval, int dimensionality, size_t *tokenCount,
    bool doMean, bool atlas
) {
    (void)texts;
    (void)embeddings;
    (void)isRetrieval;
    (void)dimensionality;
    (void)tokenCount;
    (void)doMean;
    (void)atlas;
    throw std::logic_error(std::string(implementation().modelType()) + " does not support embeddings");
}
