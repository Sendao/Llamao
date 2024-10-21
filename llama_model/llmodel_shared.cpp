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

int LLModel::pickNextTalker(  PromptContext &parentCtx, std::string username, std::string lastTalker, std::vector<std::string> actorNames )
{
    std::string query = "Who should talk next? (";
    std::string framing = "It should be ";
    std::vector<std::string>::iterator it;
    std::unordered_map<std::string, int> pollData;
    int iUser=-1, iFirst=-1;
    std::string firstActorName;

    int i=0;
    for( it = actorNames.begin(); it != actorNames.end(); it++, i++ ) {
        if( *it == "System" ) continue;
        if( *it == username ) {
            iUser = i;
        }
        if( *it == lastTalker ) continue;

        pollData[*it] = i;

        if( iFirst == -1 ) {
            firstActorName = *it;
            iFirst = i;
            pollData["I"] = i;
            pollData["Me"] = i;
        } else {
            query.append("/");
        }

        query.append(*it);
    }
    query.append(")");
    if( pollData.size() == 0 ) {
        return iUser;
    }
    if( pollData.size() == 3 ) {
        for( auto &pair : pollData ) {
            return pair.second;
        }
    }
    //! todo: ask all active actors and compare results
    return selectAnswer(firstActorName, query, parentCtx, pollData, framing);
}

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
    int ires = generateResponse3(pctx, actor, actor, n_last_batch, answers);
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
    std::string res = generateResponse2(pctx, actor, actor, n_last_batch, NULL);
    std::cerr << "queryActor generate complete: " + res + "\n";
    rewindToMark();

    return res;
}

void LLModel::runQuery( std::string who, std::string key, std::string query, std::string frame,
                        PromptContext &parentCtx,
                        std::function<bool(int32_t, const std::string&, int, int, float *, float *)> responseCallback,
                        bool forgetAboutIt
                       )
{
    std::string formed = "<|im_start|>System\n" + query + "<|im_end|><|im_start|>" + who + "\n";
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

    if( forgetAboutIt )
        markRewind();

    responseCallback(-1, formed, 0, 0, NULL, NULL);

    markGeneration(who);
    int n_last_batch = decodePrompt2("System", who, formed);
    std::cerr << "queryActor decode complete\n";
    std::string res = generateResponse2(pctx, who, who, n_last_batch, responseCallback);
    std::vector<int> tokens;
    rewindGeneration(res, tokens);

    std::cerr << "queryActor generate complete: " + res + "\n";

    if( forgetAboutIt )
        rewindToMark();

    if( key != "" ) {
        setKey(who, key, res);
    }

    return;
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
    std::vector<std::string>::iterator it;
    std::string firstActorName;
    std::vector<std::string> actorNames;

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

    if( prompt[0] == '?' ) { // ?key[*actor][&framing]:query
        std::string key = "";
        std::string query = "";
        std::string who = "";
        std::string frame = "";

        int seppos = prompt.find(":");
        if( seppos == std::string::npos ) {
            queryActorNames(actorNames);
            for( it = actorNames.begin(); it != actorNames.end(); it++) {
                if( *it == "System" ) continue;
                who = *it;
                break;
            }
            query = prompt.substr(1);
            runQuery( who, "", query, "", promptCtx, responseCallback );
            return;
        }
        int starpos = prompt.find("*");
        int andpos = prompt.find("&");
        if( andpos != std::string::npos ) {
            frame = prompt.substr(andpos+1, seppos-1);
            prompt = prompt.substr(0,andpos) + prompt.substr(seppos+1);
        }
        if( starpos > seppos || starpos == std::string::npos ) {
            queryActorNames(actorNames);
            for( it = actorNames.begin(); it != actorNames.end(); it++) {
                if( *it == "System" ) continue;
                who = *it;
                break;
            }
        }
        key = prompt.substr(1, seppos-1);
        query = prompt.substr(seppos+1);

        runQuery( who, key, query, frame, promptCtx, responseCallback );
        return;
    }

    if( prompt[0] == '^' ) { // ^actor&framing:query || ^actor:query || ^&framing:query
        std::string query = prompt.substr(1);
        std::string who, frame="";

        int seppos = prompt.find(":");
        if( seppos == std::string::npos ) {
            queryActorNames(actorNames);
            for( it = actorNames.begin(); it != actorNames.end(); it++) {
                if( *it == "System" ) continue;
                who = *it;
                break;
            }
            runQuery( who, "", query, "", promptCtx, responseCallback );
            return;
        }
        int andpos = prompt.find("&");
        if( andpos != std::string::npos ) {
            frame = prompt.substr(andpos+1, seppos-1);
            prompt = prompt.substr(0,andpos) + prompt.substr(seppos+1);
        }
        query = prompt.substr(seppos+1);

        runQuery( who, "", query, frame, promptCtx, responseCallback);
        return;
    }

    if( prompt[0] == '#' ) { // #actor&framing:query || ^actor:query || ^&framing:query
        std::string query = prompt.substr(1);
        std::string who, frame="";

        int seppos = prompt.find(":");
        if( seppos == std::string::npos ) {
            queryActorNames(actorNames);
            for( it = actorNames.begin(); it != actorNames.end(); it++) {
                if( *it == "System" ) continue;
                who = *it;
                break;
            }
            query = prompt.substr(1);
            runQuery( who, "", query, "", promptCtx, responseCallback );
            return;
        }
        int andpos = prompt.find("&");
        if( andpos != std::string::npos ) {
            frame = prompt.substr(andpos+1, seppos-1);
            prompt = prompt.substr(0,andpos) + prompt.substr(seppos+1);
        }
        query = prompt.substr(seppos+1);

        runQuery( who, "", query, frame, promptCtx, responseCallback, false);
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
    // decode the user prompt

    std::vector<int> tokens;
    int n_last_batch = decodePrompt2(fromname, toname, prompt);

    queryActorNames(actorNames);
    actorNames.push_back(fromname);

    int iName;
    std::string newprompt;
    std::string lastActor=fromname;
    while( true ) {
        tokens.clear();
        if( actorNames.size() == 3 ) { // there's only one actor to pick.
            if( lastActor == fromname ) {
                iName = 1;
            } else {
                iName = 2;
            }
        } else {
            std::cerr << "pick actor...\n";
            iName = pickNextTalker(promptCtx, fromname, lastActor, actorNames);
            std::cerr << "pick actor " << toname << "\n";
        }
        toname = actorNames[iName];
        if( toname == fromname ) {
            std::cerr << "now it's the user's turn.\n";
            break;
        }
        std::string msgbuf = "<|im_start|>" + toname + "\n";
        tokens.clear();

        markGeneration(toname);
        decodePrompt2(toname, toname, msgbuf);
        // here we are generating the response so the toname is the same as the fromname.
        newprompt=generateResponse(responseCallback, promptCtx, toname, toname, n_last_batch, tokens);
        // rewind_generation will proceed with sending the message to 'all'.
        rewindGeneration(msgbuf + newprompt, tokens);

        lastActor=toname;
        std::cerr << "done [one line]\n";
    };
}

int LLModel::decodePrompt(std::function<bool(int32_t, int, int, float*, float*)> promptCallback,
                           std::function<bool(int32_t, const std::string&, int, int, float*, float*)> responseCallback,
                           PromptContext &promptCtx,
                           //std::vector<Token> tokens, //embd_inp,
                           std::string fromname,
                           std::string toname,
                           std::string prompt,
                           std::vector<int> &tokens) {
    // save the context size
    promptCtx.n_ctx = contextLength();
    return evalTokens(prompt, tokens, fromname, toname);
}
int LLModel::decodePrompt2(std::string fromname,
                          std::string toname,
                          std::string prompt) {
    std::vector<int> tokens;
    return evalTokens(prompt, tokens, fromname, toname);
}


std::string LLModel::generateResponse(std::function<bool(int32_t, const std::string&, int, int, float*, float*)> responseCallback,
                               PromptContext &promptCtx, std::string fromname, std::string toname,
                               int n_last_batch, std::vector<int> &tokens) {
    std::string end_literal = "<|im_end|>";
    std::string new_literal = "<|im_start|>";
    std::string fullResponse;

    // predict next tokens
    int32_t n_gen=promptCtx.n_predict;
    if( n_gen == 0 ) n_gen = 1024;
    std::string buf = "", prebuf = "";
    bool found;
    std::vector<int>::iterator it;
    bool ending=false;
    bool gen_new=false;
    bool finished_gen=false;
    std::string activename=fromname;
    bool sendToAll;
    const char *p1, *p2, *pbuf, *pbufstart;
    uint16_t ptr;

    sendToAll = ( toname == "all" );

    std::cerr << "genResponse(" << fromname << ")\n";
    while( true ) {
        std::cerr << "tokens.size() = " << promptCtx.tokens.size() << "\n";
        feedData( promptCtx.logits, promptCtx.embds );
        auto id = sampleToken(promptCtx, n_last_batch);
        const std::string str = tokenToString(id);

        if( (n_last_batch=evalTokens(str, tokens, activename, toname)) == 0 ) {
            std::cerr << implementation().modelType() << " ERROR: Failed to predict next token\n";
            id = 32000; // end
        }
        promptCtx.tokens.emplace_back( id );
        buf += std::string(str);
        fullResponse += std::string(str);

        auto mlogits = promptCtx.logits;
        auto membd = promptCtx.embds;

        found = false;
        pbufstart = pbuf = buf.c_str();
        prebuf = "";
        for( p1 = end_literal.c_str(), p2 = pbuf; *p1 && *p2; p2++ ) {
            if( *p1 == *p2 ) {
                p1++;
                if( !found ) {
                    found=true;
                    pbuf = p2;
                }
            } else if( found ) {
                found=false;
                p1 = end_literal.c_str();
            }
        }
        if( found ) {
            ptr = pbuf-pbufstart;
            prebuf = buf.substr(0,ptr);
            buf = buf.substr(ptr,buf.length()-ptr);
            if( prebuf.length() > 0 ) {
                if (!responseCallback(id, std::string(prebuf), mlogits.size(), membd.size(), mlogits.data(), membd.data())) {
                    promptCtx.n_predict = 0;
                    std::cerr << "Generation aborted by responseCallback.\n";
                    break;
                }
            }
            if( buf.ends_with(end_literal) )
                break;
            continue;
        }

        // share with cb
        if (!responseCallback(id, std::string(buf), mlogits.size(), membd.size(), mlogits.data(), membd.data())) {
            promptCtx.n_predict = 0;
            std::cerr << "Generation aborted by responseCallback.\n";
            break;
        }
        buf = "";
    }

    return fullResponse;
}
std::string LLModel::generateResponse2(PromptContext &promptCtx, std::string fromname, std::string toname,
                            int n_last_batch,
                            std::function<bool(int32_t, const std::string&, int, int, float *, float *)> responseCallback
) {
    std::string end_literal = "<|im_end|>";
    std::string buf = "", prebuf = "";
    std::string res;
    std::vector<int>::iterator it;
    bool found;
    const char *p1, *p2, *pbuf, *pbufstart;
    uint16_t ptr;
    std::vector<int> newTokens;

    std::cerr << "genResponse2(" << fromname << "," << toname << ")\n";
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

        found = false;
        pbufstart = pbuf = buf.c_str();
        prebuf = "";
        for( p1 = end_literal.c_str(), p2 = pbuf; *p1 && *p2; p2++ ) {
            if( *p1 == *p2 ) {
                p1++;
                if( !found ) {
                    found=true;
                    pbuf = p2;
                }
            } else if( found ) {
                found=false;
                p1 = end_literal.c_str();
            }
        }
        if( found ) {
            uint16_t ptr = pbuf-pbufstart;
            prebuf = buf.substr(0,ptr);
            buf = buf.substr(ptr,buf.length()-ptr);
            if( prebuf.length() > 0 ) {
                if ( responseCallback != NULL && !responseCallback(-1, std::string(prebuf), 0, 0, NULL, NULL) ) {
                    std::cerr << "Generation aborted by responseCallback.\n";
                    break;
                }
            }
            if( buf.ends_with(end_literal) )
                break;
            continue;
        }

        // share with cb
        if ( responseCallback != NULL && !responseCallback(-1, std::string(buf), 0, 0, NULL, NULL) ) {
            std::cerr << "Generation aborted by responseCallback.\n";
            break;
        }
        buf = "";
    }

    return res;
}
int LLModel::generateResponse3(PromptContext &promptCtx, std::string fromname, std::string toname, int n_last_batch,
                                std::unordered_map< std::string, int > &answers)
{
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
