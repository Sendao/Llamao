#include "llmodel.h"

//#include <cassert>
#include <iostream>
#include <string>

/*
 * - we will do this but first we need to refine the token sampler to return probabilities properly
 *
int LLModel::pollAnswer( std::string query, PromptContext &parentCtx, std::vector<std::string> &answers )
{
    std::vector<std::string> actorNames;
    queryActorNames(actorNames);
    std::vector<std::string>::iterator it;
    std::string actorName;
    std::vector<int> results;

    for( it = actorNames.begin(); it != actorNames.end(); it++ ) {
        actorName = *it;

    }
}
*/

int LLModel::selectAnswer( std::string actor, std::string query, PromptContext &parentCtx,
                          std::unordered_map<std::string, int> &answers, std::string framing )
{
    std::string formed = "<|im_start|>System\n" + query + "<|im_end|><|im_start|>" + actor + "\n" + framing;
    PromptContext pctx;
    pctx.n_batch = 64;
    pctx.n_last_batch_tokens = 0;
    pctx.min_p = parentCtx.min_p;
    pctx.repeat_last_n = 64;
    pctx.top_k = parentCtx.top_k;
    pctx.top_p = parentCtx.top_p;
    pctx.temp = parentCtx.temp;
    pctx.repeat_penalty = parentCtx.repeat_penalty;
    pctx.contextErase = parentCtx.contextErase;

    markRewind();
    int n_last_batch = decodePrompt2("System", actor, formed);
    std::cerr << "selectAnswer decode complete\n";
    int ires = generateResponse3(pctx, "System", actor, n_last_batch, answers);
    std::cerr << "selectAnswer generate complete\n";
    rewindToMark();

    return ires;
}

std::string LLModel::queryActor( std::string actor, std::string query, PromptContext &parentCtx )
{
    std::string formed = "<|im_start|>System\n" + query + "<|im_end|><|im_start|>" + actor + "\n";
    PromptContext pctx;
    pctx.n_batch = 64;
    pctx.n_last_batch_tokens = 0;
    pctx.min_p = parentCtx.min_p;
    pctx.repeat_last_n = 64;
    pctx.top_k = parentCtx.top_k;
    pctx.top_p = parentCtx.top_p;
    pctx.temp = parentCtx.temp;
    pctx.repeat_penalty = parentCtx.repeat_penalty;
    pctx.contextErase = parentCtx.contextErase;

    markRewind();
    int n_last_batch = decodePrompt2("System", actor, formed);
    std::cerr << "queryActor decode complete\n";
    std::string res = generateResponse2(pctx, "System", actor, n_last_batch);
    std::cerr << "queryActor generate complete: " + res + "\n";
    rewindToMark();

    return res;
}

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

    std::vector<std::string> actorNames;
    queryActorNames(actorNames);
    actorNames.push_back(fromname);

    std::string query = "Who should reply?";
    std::string framing = "It should be ";
    std::vector<std::string>::iterator it;
    bool started=false;
    std::unordered_map<std::string, int> pollData;

    int i=0;
    for( it = actorNames.begin(); it != actorNames.end(); it++ ) {
        if( started ) query.append("/");
        started=true;
        query.append(*it);
        pollData[*it] = i;
        if( *it == toname ) {
            pollData["I"] = i;
            pollData["Me"] = i;
        }
        i++;
    }
    query.append(")");

    int iName;
    while( true ) {
        iName = selectAnswer(toname, query, promptCtx, pollData, framing);
        toname = actorNames[iName];
        std::cerr << "pick actor " << toname << "\n";
        if( toname == fromname ) {
            std::cerr << "now it's the user's turn.\n";
            break;
        } else {
            std::string msgbuf = "<|im_start|>" + toname + "\n";
            decodePrompt(promptCallback, responseCallback, promptCtx, toname, "all", msgbuf);
        }
        markRewind();
        generateResponse(responseCallback, promptCtx, fromname, toname, n_last_batch);
        rewindToMark();
        decodePrompt(promptCallback, responseCallback, promptCtx, toname, "all", prompt);
    };
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
int LLModel::decodePrompt2(std::string fromname,
                          std::string toname,
                          std::string prompt) {
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
        auto id = sampleToken(promptCtx, n_last_batch);
        newTokens.clear();
        newTokens.push_back(id);
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

        // share with cb
        if (!responseCallback(id, std::string(str), mlogits.size(), membd.size(), mlogits.data(), membd.data())) {
            promptCtx.n_predict = 0;
            i = 0;
            std::cerr << "Generation aborted by responseCallback.\n";
            break;
        }

        if( generatedTokens > 0 && buf.ends_with(end_literal) ) {
            break;
        }
        if( !end_literal.starts_with(buf) ) {
            const char *p1, *p2;
            int chars=0;
            bool found=false;
            for( p1 = end_literal.c_str(), p2 = buf.c_str(); *p1 && *p2; p2++ ) {
                if( *p2 == *p1 ) {
                    found=true;
                    p1++;
                    chars++;
                } else if( found ) {
                    found=false;
                    chars=0;
                    p1 = end_literal.c_str();
                }
            }
            if( !found ) {
                generatedTokens++;
                buf="";
            } else {
                char buf1[24], *p3;
                for( p3=buf1, p2 = end_literal.c_str(); *p2 && chars>0; p2++, p3++, chars-- ) {
                    *p3 = *p2;
                }
                *p3 = '\0';
                buf = buf1;
                std::cerr << "Reduce buffer to '" << buf1 << "'\n";
            }
        }
    }
}
std::string LLModel::generateResponse2(PromptContext &promptCtx, std::string fromname, std::string toname, int n_last_batch) {
    std::string cachedResponse;
    std::string end_literal = "<|im_end|>";
    int i;
    std::string buf = "";
    std::string res;
    std::vector<int>::iterator it;

    int generatedTokens=0;
    std::vector<int> newTokens;

    std::cerr << "genResponse2()\n";
    while( true ) {
        feedData( promptCtx.logits, promptCtx.embds );
        std::cerr << "sampleToken n_last_batch=" << n_last_batch << "\n";
        auto id = sampleToken(promptCtx, n_last_batch);
        newTokens.clear();
        newTokens.push_back(id);

        const std::string str = tokenToString(id);
        std::cerr << "gen: " << str << "(" << id << ")\n";
        if( (n_last_batch=evalTokens(str, newTokens, toname, fromname)) == 0 ) {
            std::cerr << implementation().modelType() << " ERROR: Failed to predict next token\n";
            id = 32000; // end
            buf += "<|im_end|>";
        } else {
            buf += std::string(str);
        }
        promptCtx.tokens.emplace_back( id );

        if( buf.find(end_literal) != std::string::npos ) {
            break;
        }
        if( !end_literal.starts_with(buf) ) {
            generatedTokens++;
            res += buf;
            buf = "";
        }
    }

    return res;
}
int LLModel::generateResponse3(PromptContext &promptCtx, std::string fromname, std::string toname, int n_last_batch,
                                std::unordered_map< std::string, int > &answers)
{
    std::string cachedResponse;
    std::string end_literal = "<|im_end|>";
    int i;

    std::string buf = "", answerBuf;
    std::vector<std::string>::iterator sit;
    std::vector<int> newTokens;
    std::unordered_map<int, int> possible;

    int invalid_tokens=0, max_invalid=32;

    int selected_answer=0;

    std::cerr << "genResponse3()\n";
    while( true ) {
        feedData( promptCtx.logits, promptCtx.embds );
        std::cerr << "sampleToken n_last_batch=" << n_last_batch << "\n";
        /*
        auto id = sampleToken(promptCtx, n_last_batch);
        newTokens.clear();
        newTokens.push_back(id);
        std::cerr << "tokenToString(" << id << ")\n";
        const std::string str = tokenToString(id);
        std::cerr << "gen: " << str << "(" << id << ")\n";
        if( (n_last_batch=evalTokens(str, newTokens, fromname, toname)) == 0 ) {
            std::cerr << implementation().modelType() << " ERROR: Failed to predict next token\n";
            id = 32000; // end
        }
        promptCtx.tokens.emplace_back( id );
        buf += std::string(str);
        possible.clear();

        int newresult, resultSize=0;
        bool found;

        for( const auto &pair : answers ) {
            answerBuf = pair.second;
            if( answerBuf.starts_with(buf) ) {
                found=false;
                for( const auto &pair2 : possible ) {
                    if( pair2.first == pair.first ) {
                        newresult = pair2.second+1;
                        found=true;
                        break;
                    }
                }
                if( !found ) {
                    possible[pair.first] = 1;
                } else {
                    possible[pair.first] = newresult;
                }
            }
        }
        if( possible.size() == 1 ) {
            for( const auto &pair : possible ) {
                selected_answer = pair.first;
                break;
            }
            break;
        }
        if( possible.size() == 0 ) {
            invalid_tokens++;
            if( invalid_tokens >= max_invalid ) {
                std::cerr << "Invalid answers. Defaulting to answer 0.\n";
                break;
            }
            std::cerr << "Invalid answer [" << buf << "].\n";
            buf = "";
        }
        */
        selected_answer = pollVocab( answers, promptCtx.logits.data() );
        std::cerr << "Got answer: " << selected_answer << "\n";

        return selected_answer;

        //if( buf.ends_with(end_literal) ) {
        //    break;
        //}
    }
    /*
    std::cerr << "Got answer: " << selected_answer << ": " << answers[selected_answer] << "\n";
    return selected_answer;
    */
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
