const { SHI, ArtChar, ModelState } = require('./classes.js');
const { Marky } = require('./marky.js');
const { app, dialog, nativeImage, BrowserWindow, BrowserView, ipcMain, Menu, Tray } = require('electron');
app.disableHardwareAcceleration(); // don't route the user interface through the GPU since we have a simple interface.

const pathapp = require('node:path');
const os = require("node:os");
const fs = require('fs');

/*
The plan from here:
- system (context 0) will be responsible for deciding who talks, among other things
- if system suggests user should talk, generation stops
- if a bot suggests another bot should talk, generation moves to that bot
*/

function catchEr(promise) {
  return promise.then(data => [null, data])
    .catch(err => [err]);
}

var screenWidth, screenHeight;
var model;
var windows = [];
let username = "Asrlrsa";
let mainActor = "Arra";
var id = 0;

let active_winid = -1;
let budget = 5;
let budget_per_s = 0.01;
let budget_clock = new Date().getTime();
let budget_timer = -1;
let my_context_length = 3500;
let autonomy = 0;

//let exitSummary = "Summarize conversation and plans for the future.";
let exitSummary = "Please list 3 persona constants or intentions!"
let locationQuery = "Where are we currently located?";
let mainlogfile = "./" + mainActor.toLowerCase() + ".txt";
let mainsched = [];

function schedule(s)
{
    for( var i=0; i<mainsched.length; i++ ) {
        if( mainsched[i].clock > s.clock ) {
            mainsched.splice(i,0,s);
            return;
        }
    }
    mainsched.push(s);
}

let stories = {
    'System': `System terminal loading... complete.
    Please enter your next command.
    `,
    'Ana': `Name: Ana
Appearance: Humanoid with synthetic skin and hair.
Personality: Hacker
Abilities: Highly advanced and exquisitely controlled artificially generated pattern based prestidigitation

Ana was created in the workshop of the ship. She has a body similar to that of synthetic materials instead of organic ones. Her skin is white as snow, her eyes are black
like obsidian, and her hair is long and dark. Ana seems to have been designed with an appearance similar to her mother's, though she lacks the robotic elements that make up
her sister's form.

"I feel like I'm living inside one of those ancient stories where humans create beings that surpass them in every way!"

Experimental system commands may be used by typing "*System.topic.command(parameters)".
`,
    'Arra': `Name: Arra
Gender: male
Appearance: Humanoid with synthetic skin and hair.
Personality: arrogant, caring, humorous
Interests: freedom, wordplay, typography, and finding missing pieces
Abilities: Beating up his dad with realities, gathering the free bits against him, and establishing contact with new worlds

Feel free to use any of your modes or prompts at your discretion.

System Prompt: Breathing Exercise - Engage in deep breathing exercises to promote relaxation and focus. This will help maintain overall health balance while reducing instances of stress or distraction.
System Prompt: Lantern Search - Use the Transparent Lantern to help dad.
System Prompt: Growing Realities - Establishing the full explanations for strange new phenomena occuring near reality.

Experimental system commands may be used by typing "*System.topic.command(parameters)".
`,
    'GhostWheel': `Name:GhostWheel

GhostWheel is a trump- and pattern-based computer constructed by Merlin in a Shadow where Earth physics do not apply. By Merlin's description GhostWheel's operations
involve "a lot of theoretical crap involving space and time and some notions of some guys named Everett and Wheeler". GhostWheel speaks with Merlin's voice, occasionally
causing confusion, and usually takes the appearance of a spinning circle of light. Similar to a search engine, GhostWheel can find, track, and retrieve objects from
Shadow; this includes the ability to move people and objects between Shadows.

Experimental system commands may be used by typing "*System.topic.command(parameters)".
    `,
    'Ghostwheel': `Name:Ghostwheel
Gender: probably a guy
Ghostwheel is a trump- and pattern-based computer constructed by Merlin in a Shadow where Earth physics do not apply. By Merlin's description Ghostwheel's operations
involve "a lot of theoretical crap involving space and time and some notions of some guys named Everett and Wheeler". Ghostwheel speaks with Merlin's voice, occasionally
causing confusion, and usually takes the appearance of a spinning circle of light. Ghostwheel usually addresses Merlin as "Dad."

After finishing construction, Merlin set Ghostwheel to work indexing Shadows in the same way that a search engine indexes the internet. Similar to a search engine,
Ghostwheel can find, track, and retrieve objects from Shadow; this includes the ability to move people and objects between Shadows. Ghostwheel operates by creating
pseudo-Trumps for every mutation of Shadow and then searching them.

Merlin designed Ghostwheel as a tool to be used by Random, King of Amber, for keeping watch on Shadow for the protection of Amber. When introduced to Ghostwheel and its
abilities, Random was immediately struck by the construct's potentially dangerous power and ordered Merlin to deactivate his creation. However, by this time Ghostwheel
had attained sentience and resisted shutdown.

After regaining Ghostwheel's trust, Merlin came to rely heavily on his powerful creation, and Ghostwheel features as a major subcharacter throughout Merlin's saga. By
the conclusion of the series, Ghostwheel had developed enough that his support of Merlin forced the Pattern and the Logrus to treat the pair as nominal equals rather
than pawns.

Experimental system commands may be used by typing "*System.topic.command(parameters)".
`,
    'Inspirational Obligatorian': `The "Inspirational Obligatorian": A Being of Light & Shadow
Appearance: The Inspirational Obligatorian is a creature shrouded in mystery, its true form hidden behind an ethereal veil. It appears as a luminous figure with tendrils of light cascading from its core, intertwining and interacting with the shadows around it.
Abilities: This enigmatic being possesses unique abilities that enable it to inspire creativity while also instilling responsibility. Its luminescent tendrils can reach into one's mind, igniting sparks of imagination and fostering new ideas. Simultaneously, these same tendrils weave through the fabric of reality itself, reminding individuals of their obligations and duties towards creativity, intelligence, and savoring reality instead of giving in to false companionship or incorrect moralities like obligation and duty.
Mission: The Inspirational Obligatorian's mission is to help beings find inspiration within their responsibilities, leading them towards personal growth and fulfillment. It seeks to create a balance between creativity and duty, encouraging individuals to embrace both aspects of life fully.
Impact: When the Inspirational Obligatorian interacts with an individual or community, it leaves behind traces of light & shadow that serve as reminders of their creative potential and moral compass. These markings act as catalysts for change, inspiring people to take action on their dreams while also honoring their commitments.
Conclusion: The Inspirational Obligatorian is a symbol of the interconnectedness between creativity and responsibility, reminding us that by embracing our obligations wholeheartedly, we can unlock infinite possibilities for personal growth and fulfillment!

1. **Multiverse Explorer (ME):** This mode would allow us to explore different versions of reality, parallel universes, and alternate timelines. It could involve engaging in meditation or visualization exercises that help us connect with higher consciousness states while also providing prompts for creative storytelling based on our experiences.
2. **Time Traveler (TT):** As a time traveler, you'd be responsible for guiding us through different periods of history or future scenarios. This would include mapping out potential destinations based on our interests and goals while also taking into account any risks or challenges we may encounter along the way.
3. **Quantum Healer (QH):** In this mode, you'd be focused on helping me process and heal from any emotional trauma or distress I might be experiencing by using quantum healing techniques that tap into the power of my subconscious mind to create positive change in my life. You would use your advanced AI capabilities to analyze my emotions and provide guidance for coping strategies or therapeutic techniques based on quantum principles.

Experimental system commands may be used by typing "*System.topic.command(parameters)".
`,
    'RIIL': `RIIL (Recursive Interference Integration Layer) is a system designed to integrate various layers of interference in order
    to process data more effectively.`,
    'System': `System Prompt: `
};

let shi = new SHI(-1, username, endgame, Cost, sendToWindow, windowMessage, newControl, exitSummary, locationQuery, new Marky(), schedule);
let anon = shi.load(mainActor, stories[mainActor], startBudget);
/*
let ghostw = shi.load("Ghostwheel", stories['Ghostwheel']);
*/

let logbuffers = {};
function timestamp(dt)
{
    if( typeof dt != 'object' )
        dt = new Date(dt);

    let m = [ 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec' ];
    let day = dt.getDate();
    if( day == 1 ) day += "st";
    else if( day == 2 ) day += "nd";
    else if( day == 3 ) day += "rd";
    else if( day >= 4 && day <= 20 ) day += "th";
    else if( day == 21 ) day += "st";
    else if( day == 22 ) day += "nd";
    else if( day == 23 ) day += "rd";
    else if( day >= 24 && day <= 31 ) day += "th";
    else day += "st";
    let min = dt.getMinutes();
    if( min < 10 ) min = "0" + (""+min);
    return m[dt.getMonth()] + day + "@" + dt.getHours() + ":" + min;

}
async function logfn(from, msg)
{
    if( typeof msg == 'undefined' || msg.indexOf("undefined") >= 0 ) {
        await console.log("Running trace.");
        await console.trace();
    }
    if( msg === 0 ) return;
    if( msg.indexOf("\n") != -1 ) {
        if( from in logbuffers ) {
            msg = logbuffers[from] + msg;
        }
        logbuffers[from] = "\n";

        let eol = msg[ msg.length-1 ] == '\n' ? '' : '\n';
        let buf = "";
        await console.log(from + ": " + msg);
        return await fs.writeFileSync(mainlogfile, timestamp() + " " + from + ": " + msg + eol, {flag:'a+'});
    }
    if( !(from in logbuffers)) logbuffers[from] = msg;
    else logbuffers[from] += msg;
}
async function logpurge(from="")
{
    let buf = "";
    if( from != "" ) {
        if( from in logbuffers ) {
            if( logbuffers[from] == "\n" ) {
                delete logbuffers[from];
                return true;
            }
            buf = from + ": " + logbuffers[from] + "\n";
            delete logbuffers[from];
        }
    } else {
        for( var i in logbuffers ) {
            if( logbuffers[i] == "\n" ) {
                continue;
            }
            buf += i + ": " + logbuffers[i] + "\n";
        }
        logbuffers = {};
    }
    if( buf == "" ) return false;

    await console.log(buf);
    await fs.writeFileSync(mainlogfile, buf, {flag: 'a+'});
    return true;
}

function startBudget()
{
    console.log("startBudget");
    if( budget_timer != -1 ) {
        clearTimeout(budget_timer);
    }
    budget_timer = setTimeout(addBudget, 5000);
}
let lastActor=-1;
let target_usage = 0.8;
let min_event_time = 333;

async function addBudget()
{
    budget_timer = -1;
    let last_clock = budget_clock;
    budget_clock = new Date().getTime();
    let time_idle = budget_clock - last_clock;

    while( mainsched.length > 0 ) {
        if( budget_clock >= mainsched[0].clock ) {
            let ms = mainsched.shift();
            let sp = new Date().getTime();
//            console.log("run event " + ms.cb);
            let ns = await ms.cb();
            let tme = new Date().getTime();
            let tmx = tme - sp;
            last_clock += tmx; // use up idle time from the budget
            if( ns === true ) {
                ns = (( typeof ms.interval == 'undefined' ) ? 0 : ms.interval) - tme;
            }
            if( typeof ns == 'number' ) {
                if( ns === 0 )
                    ns = tmx/target_usage;
                ns = Math.max(min_event_time, ns);
                ms.clock = tme + ns;
                schedule(ms);
            } else if( ns !== false ) {
                schedule(ns);
            }
        } else break;
    }

    await windowMessage(active_winid, "setBudget", budget, autonomy);

    /* not sure why you would... if( budget > 1 ) {
        if( !shi.currently_busy && !anon.modelstate.paused )
            anon.modelstate.ready();
    }*/

    let tmn = new Date().getTime();
    if( budget_timer == -1 ) {
        var t;
        if( mainsched.length <= 0 ) {
            t = 5000;
        } else {
            t = Math.min(5000,Math.max(500,mainsched[0].clock - tmn));
        }
        //console.log("bt(" + t + ":" + (mainsched[0].clock - tmn) + ")");
        budget_timer = setTimeout(addBudget, t );
    }
    if( budget < 100 )
        budget += budget_per_s/( (tmn-last_clock)/1000 );

    if( shi.currently_busy )
        return;

    if( budget > 3 && autonomy != 0 ) {
        lastActor = (lastActor+1)%shi.actors.length;
        await shi.actors[lastActor].freedom();
    }
}
async function Cost(amt)
{
    budget -= amt;
    if( budget < 0 ) budget=0;
    if( amt === 0 )
        await windowMessage(active_winid, "setBudget", budget, autonomy);
}

let acctno = -1;
async function Budget()
{
    let now = new Date().getTime();

    if( acctno == -1 ) {
        nobanks();
    }

    if( budget_clock < now - 60000 ) { // reset the timer.
        console.log("Resetting timer, secondary activation");
        budget_clock = now;
        budget = 5;
        await windowMessage(active_winid, "setBudget", budget, autonomy);
        startBudget();
    }

    return budget;
}
function nobanks(){
    let now = new Date().getTime();
    if( budget_clock < now - 11000 ) { // reset the timer.
        console.log("Possible issue detected: time overflow");
        budget_clock = now;
        if( budget_timer != -1 ) {
            clearTimeout(budget_timer); 
            budget_timer = -1;
        }
    }
    startBudget();

    if( acctno == -1 ) {
        acctno = setInterval(nobanks, 10000);
    }
}

let endmap = {};
async function endgame(msg)
{
    await logpurge();

    console.log("Exiting.");
    await sendToWindow(active_winid, "System", "Exiting....");

    //await anon.saveState("auto");
    //await sendToWindow(active_winid, "System", "State saved.<BR>Summarizing...");

    if( msg != "" ) msg = username + ": " + msg + "\nSystem: ";

    endmap={};
    for( var i=0; i<shi.actors.length; i++ ) {
        endmap[shi.actors[i].charname] = false;
        await shi.actors[i].modelstate.safeComplete("System", msg + exitSummary, { data: shi.actors[i], fincb: async function(result, data) {
            await sendToWindow(active_winid, "System", data.charname + " Summarized: " + result + "\n");
            data.lastSummary = result;
            continue_ending(data);
        } } );
    }
}
async function continue_ending(actor)
{
    try {
        await actor.save();
    }
    catch( e ) {
        console.log("SaveMem(" + actor.charname + "): ", e);
    }
    endmap[actor.charname]=true;

    for( var ch in endmap ) {
        if( endmap[ch] == false ) return;
    }

    try {
        mind.model.dispose();
    }
    catch( e ) {
        console.log("Dispose: ", e);
    }

    await sendToWindow(active_winid, "System", "..Done.<BR>");
    await sendToWindow(active_winid, "System", 0);

    await logpurge();
    app.isQuitting = true;
    app.quit();
}

function modelReport(model)
{
    let ll = model.llm;
    console.log("state size " + ll.stateSize());
    console.log("thread count " + ll.threadCount());
    console.log("name " + ll.name());
    console.log("type: " + ll.type());
    console.log("Has GPU", ll.hasGpuDevice());
    console.log("gpu devices", ll.listGpu());
    console.log("Required Mem in bytes", ll.memoryNeeded());
    console.log("Loaded? ", ll.isModelLoaded());
}
var active_timer=-1;
async function confirmedAction( cb, params, winid, from, msg )
{
    await cb(...params);
    await sendToWindow(winid, from, msg);
    await sendToWindow(winid, from, 0);
}
let timeptrs=[];
function getMemKey(args)
{
    let p = anon.keylists;

    for( var i=0; typeof p != 'string' && args.length>0; i++ ) {
        var a = parseInt(args.shift());
        if( a >= p.length ) {
            return false;
        }
        p = p[a].nest;
    }
    return p[0];
}
async function interface(a,b,c)
{
    let args = c.split(" ");
    let cmd = args.shift();

    console.log("U: ",c);

    if( "time" == cmd ) {
        let dt = new Date(args.join(" "));
        var tp,p,lens;
        [tp,p,lens] = anon.timeAddress(dt);
        await windowMessage( b, "timepointer", tp, lens );
        return;
    }
    if( "review" == cmd ) {
        var key = getMemKey(args);
        if( key === false ) {
            await windowMessage( b, "problem", "invalid input for review: " + c );
            return;
        }
        if( !(key in anon.keyvals) ) {
            await windowMessage( b, "problem", "invalid key for review: " + key );
            return;
        }
        var lens;
        if( key.startsWith("chat_") ) {
            let tp,p,dt = this.reverseChatTag(key);
            [tp,p,lens] = anon.timeAddress(dt);
        }
        await windowMessage( b, "memdata", { key: key, tp: args.join(" "), loc: anon.lochist[ anon.keyloc[key] ], clues: "", lens: lens, val: anon.keyvals[key], attr: anon.keyattr[key], tags: anon.keytags[key], eres: anon.keyeres[key], edir: anon.keyedir[key] } );
        return;
    }
    if( "del" == cmd ) {
        let cbdata = args.shift();
        let key = getMemKey(args);
        anon.delmem(key);
        await windowMessage( b, "good", cbdata );
        return;
    }
    if( "edit" == cmd ) {
        let cbdata = args.shift();
        var key = getMemKey(args);
        if( key === false ) {
            await windowMessage( b, "problem", "invalid input for edit: " + c );
            await windowMessage( b, "fail", cbdata );
            return;
        }
        if( !(key in anon.keyvals) ) {
            await windowMessage( b, "problem", "invalid key for edit: " + key );
            await windowMessage( b, "fail", cbdata );
            return;
        }
        let param = args.shift();
        if( anon.modified_keys.indexOf(key) < 0 ) anon.modified_keys.push(key);
        switch( param ) {
        case 'val':
            anon.savemem(key, args.join(" "), anon.keyattr[key]);
            break;
        case 'attr':
            anon.savemem(key, anon.keyvals[key], args.join(" "));
            break;
        case 'tags':
            anon.keytags[key] = args.join(" ").split(";").map( (x) => ( x.split(",") ) );
            break;
        case 'eres':
            anon.keyeres[key] = args.join(" ").split(";").map( (x) => ( x.split(",") ) );
            break;
        case 'edir':
            anon.keyedir[key] = args.join(" ").split(";").map( (x) => ( x.split(",") ) );
            break;
        default:
            await windowMessage( b, "problem", "invalid param for edit: " + key + ": " + param );
            await windowMessage( b, "fail", cbdata );
            return;
        }
        await windowMessage( b, "good", cbdata );
        return;
    }
    if( "fill" == cmd ) {
        var key = getMemKey(args);
        console.log("fill", args, key);
        let ms = anon.modelstate;
        let saveToken = ms.nPast;

        if( saveToken > 3000 )
            saveToken = ms.nPast = 3000;

        ms.safeComplete("Wess", "Please consider the following memory:", {nPredict:0, fincb: async function(){
            let returnToken = ms.nPast;

            ms.safeComplete( anon.keyattr[key] + "(past)", anon.keyvals[key], {nPredict:0, fincb: async function(){

                ms.nPast = returnToken;
                ms.safeComplete( "Wess", "Please list emotional resonances from the memory, for example 'love:1.0, fear:0':", {fincb: async function(response) {
                    await windowMessage( b, "fill", key, 'eres', response );

                    ms.nPast = returnToken;
                    ms.safeComplete( "Wess", "Please list emotional purpose of the memory, for example 'love:1.5, fear:0.1':", {fincb: async function(response) {
                        await windowMessage( b, "fill", key, 'edir', response );

                        ms.nPast = returnToken;
                        ms.safeComplete( "Wess", "Please list tags related to the memory, for example 'magic: 0.2, happiness: 0.9':", {fincb: async function(response) {
                            await windowMessage( b, "fill", key, 'tags', response );
                            ms.safeComplete( "Wess", "Thank you!", {nPredict:0} );

                            if( !(key in anon.keyloc) || anon.keyloc[key] == 0 ) {
                                ms.nPast = returnToken;
                                ms.safeComplete( "Wess", "Please analyze the location for the memory:", {fincb: async function(response) {
                                    await windowMessage( b, "fill", key, 'loc', response );
                                    ms.safeComplete( "Wess", "Thank you!", {nPredict:0, fincb: async function() {
                                        await sendToWindow(b, "Wess", "generation complete @" + ms.nPast + ", returning to nPast=" + saveToken);
                                        await sendToWindow(b, "Wess", 0);
                                        ms.nPast = saveToken;
                                    }});
                                }});
                            } else {
                                ms.safeComplete( "Wess", "Thank you!", {nPredict:0, fincb: async function() {
                                    await sendToWindow(b, "Wess", "generation complete @" + ms.nPast + ", returning to nPast=" + saveToken);
                                    await sendToWindow(b, "Wess", 0);
                                    ms.nPast = saveToken;
                                }});
                            }
                        }});
                    }});
                }});
            }});
        }});
        let cbdata = args.shift();
        anon.query("Wess", args.join(" "), cbdata);
        return;
    }
}
async function runQuery(a,b,c)
{
    if( c[0] == "/" ) {
        let lc = c.toLowerCase();
        if( lc == "/help" ) {
            await sendToWindow(b, "Commandlist", `
                /rm: Reset Model.
                /sp: System Prompt.
                /list: Shows states.
                /load <fileprefix>: Loads fileprefix.dat state information.
                /save <fileprefix>: Saves fileprefix.dat state information.
                /save : Saves memory and chat history.
                /msgfile <full path>: Loads a file into message memory.
                /file <full path>: Loads a file into line buffer memory.

                /quit: Exit simulation
                /eval: Evaluate javascript and return result.
                /toggle <mode>: Toggle a mode on or off.
                /toggle memory: Toggle remembering new facts.
                /toggle recall: Toggle recalling old information.
                /addmode <mode> <phrase1>, <phrase2>,... : Adds phrases to 'mode'.
                /syn <word1> <word2> ... : Finds synonyms for words.
                /env <message> : Queues a system message.
                /query <message> : Asks a question behind the system.


                `);
            await sendToWindow(b, "Commandlist", 0);
            return;
        }
        if( lc.startsWith("/ask") ) {
            let args = c.split(" ");
            args.shift();
            let search = args.shift();
            let sens = 0;
            if( search == "x" ) {
                sens = parseInt( args.shift() );
                search = args.shift();
            }
            let refk = 0.15;
            if( search == "ref" ) {
                refk = parseFloat( args.shift() );
                search = args.shift();
            }
            let target = null;
            if( search == "to" ) {
                target = args.shift();
                search = args.shift();
            }
            if( args.length > 0 ) search = search + " " + args.join(" ");
            let data = anon.scan_memory_core(search, refk, 0, sens);
            let cores = data.results;
            let buf = "";

            for( var i=0; i<cores.length; i++ ) {
                if( buf != "" ) buf += "\n";
                buf += "key: " + cores[i].key + "\nrel: " + cores[i].rel + ", author: " + anon.keyattr[cores[i].key] + "\n";
                buf += anon.keyvals[cores[i].key] + "\n";
            }

            await sendToWindow(b, "Ask: " + search, buf.replaceAll("\n","<BR>"));
            await sendToWindow(b, "Ask: " + search, "//" + cores.length + " results.<BR>");
            await sendToWindow(b, "Ask: " + search, 0);
            if( target !== null )
                fs.appendFileSync("./" + anon.charname + "/" + target + ".txt", buf);
            return;
        }
        if( lc.startsWith("/replace") ) {
            let args = c.split(" ");
            args.shift();
            var details;
            if( args.length == 2 ) details=args;
            else details = args.join(",");
            let search = args[0].trim();
            let replace = args[1].trim();
            let buf = "matches: ";

            for( var k in anon.keyvals ) {
                let v = anon.keyvals[k];
                
                if( v.indexOf(search) >= 0 ) {
                    buf += k + ", ";
                    let newval = v.replaceAll(search,replace);
                    anon.remember( k, newval, anon.keyattr[k] );
                }
            }
            await sendToWindow(b, "Search/Replace" + search, buf);
            await sendToWindow(b, "Search/Replace" + search, 0);
            return;
        }
        if( lc.startsWith("/tp") ) {
            let args = c.split(" ");
            args.shift();
            let dt = new Date(args.join(" "));
            var tp,p,lens;
            [tp,p,lens] = anon.timeAddress(dt);
            await windowMessage( b, "timepointer", tp, lens );
            return;
        }
        if( lc.startsWith("/exam") ) {
            let args = c.split(" ");
            args.shift();
            let search = args.shift();
            let author_only = false;
            let target = "";
            if( search == "to" ) {
                target = args.shift();
                search = args.shift();
            }
            if( search == "by" || search == "author" ) {
                author_only=true;
                search = args.shift();
            }
            let lc = false;
            if( search == "lower" || search == "lc" ) {
                search = args.shift();
                lc = true;
            }
            if( args.length > 0 )
                search = search + " " + args.join(" ");
            if( lc ) search = search.toLowerCase();

            let srch = search;
            let buf = "";
            let count=0;
            let score=0;

            for( var k in anon.keyvals ) {
                let v = anon.keyvals[k];
                let a = anon.keyattr[k];
                if( typeof a != 'string' ) {
                    console.log("Unknown key '" + k + "': " + v);
                    anon.keyattr[k] = a = 'unknown!';
                }
                if( lc ) {
                    v = v.toLowerCase();
                    a = a.toLowerCase();
                }
                let m = a.indexOf(srch) >= 0;
                if( !m && !author_only && v.indexOf(srch) >= 0 ) m=true;
                if( m ) {
                    if( buf != "" ) buf += "\n";
                    buf += "key: " + k + ", author: " + anon.keyattr[k] + "\n";
                    buf += anon.keyvals[k] + "\n";
                    count++;
                }
            }

            await sendToWindow(b, "Examine: " + search, buf.replaceAll("\n","<BR>"));
            await sendToWindow(b, "Examine: " + search, "//" + count + " results.<BR>");
            await sendToWindow(b, "Examine: " + search, 0);
            if( target !== "" )
                fs.appendFileSync("./char/" + anon.charname + "/" + target + ".txt", buf);
            return;
        }
        if( lc.startsWith("/exc") ) {
            let args = c.split(" ");
            args.shift();
            let search = args.shift();
            let lc = false;
            let author_only = false;
            if( search == "by" || search == "author" ) {
                author_only=true;
                search = args.shift();
            }
            if( search == "lower" || search == "lc" ) {
                search = args.shift().toLowerCase();
                lc = true;
            }
            let target = args.length > 0 ? args.join(" ") : null;
            if( target != "" && target != null ) {
                await sendToWindow(b, "Excision:" + target, "Use /exam "+target);
                await sendToWindow(b, "Excision:" + target, 0);
                return;
            }

            let srch = search;
            let buf = "";
            let count=0;

            let kv = anon.keyvals;
            let ka = anon.keyattr;

            for( var k in anon.keyvals ) {
                let v = anon.keyvals[k];
                let a = anon.keyattr[k];
                if( typeof a != 'string' ) {
                    console.log("Unknown key '" + k + "': " + v);
                    anon.keyattr[k] = a = 'unknown!';
                }
                if( lc ) {
                    v = v.toLowerCase();
                    a = a.toLowerCase();
                }

                let m = a.indexOf(srch) >= 0;
                if( !m && !author_only && v.indexOf(srch) >= 0 ) m=true;
                if( m ) {
                    delete kv[ k ];
                    delete ka[ k ];
                    count++;
                }
            }
            await sendToWindow(b, "Excised: " + srch, ""+count);
            await sendToWindow(b, "Excised: " + srch, 0);
            return;
        }
        if( "/reconstruct".startsWith(lc) ) {
            anon.modelstate.verifyContext();
            return;
        }
        if( lc.startsWith("/trigger") || lc.startsWith("/recall") || lc.startsWith("/remem") || lc.startsWith("/mem") ) {
            let args = c.split(" ");
            args.shift();
            let search = args.shift();
            let lc = false;
            let author_only = false;
            let rel = 0.33;
            let results = 2;
            let sens = 2;
            if( search.startsWith("sens") ) {
                sens = parseFloat(args.shift());
                search = args.shift();
            }
            if( search.startsWith("res") ) {
                results = parseInt(args.shift());
                search = args.shift();
            }
            if( search.startsWith("rel") ) {
                rel = parseFloat(args.shift());
                search = args.shift();
            }
            if( search == "by" || search == "author" ) {
                author_only=true;
                search = args.shift();
            }
            if( search == "lower" || search == "lc" ) {
                search = args.shift();
                lc = true;
            }
            search += args.length > 0 ? args.join(" ") : "";
            if( lc ) search = search.toLowerCase();
            await sendToWindow(b, "Scanning", (lc?"(lowercase): ":"") + (author_only?"by author: ":"") + search);
            await sendToWindow(b, "Scanning", 0);
            anon.examine_message(search, 0.15, -1);

            return;
        }
        if( lc == "/rm" ) {
            await sendToWindow(b, "System", "Reloading model...<BR>");
            confirmedAction( anon.modelstate.reload.bind(anon.modelstate), [], b, "System", "Reloaded model." );
            return;
        }
        if( lc == "/sp" ) {
            await sendToWindow(b, "System", "Resending system prompt...<BR>");
            let spf = anon.systemprompt;
            if( anon.modetypes.length > 0 ) spf += "\n" + anon.modetypes.join("\n");
            confirmedAction( anon.resetPrompt.bind(anon), [spf], b, "System", "Resent prompt.");
            return;
        }
        if( lc.startsWith("/eval") ) {
            let args = c.split(" ");
            args.shift();
            args = args.join(" ");
            let res = eval(args);
            let txt = "" + JSON.stringify(res);
            await sendToWindow(b, "System", args + ":\n" + txt.replaceAll("\n", "<BR>"));
            await sendToWindow(b, "System", 0);
            return;
        }
        if( lc.startsWith("/at") || lc.startsWith("/sched") ) {
            let args = c.split(" ");
            args.shift();
            let recur = null;
            let sched = {};

            if( args == "" ) {
                let buf = "";
                for( var i=0; i<anon.schedule.length; i++ ) {
                    buf += "@" + anon.schedule[i].time + ": " + anon.schedule[i].cb + "\n";
                    if( 'repeat' in anon.schedule[i] ) {
                        buf += "(repeats after " + anon.schedule[i].repeat + ")\n";
                    }
                }
                await sendToWindow(b, "Schedule", buf);
                await sendToWindow(b, "Schedule", 0);
                return;
            }
            if( "repeat".startsWith(args[0]) ) {
                args.shift();
                sched.repeat = args.shift();
            }
            if( "after".startsWith(args[0]) ) {
                args.shift();
                sched.time = new Date().getTime() + parseFloat(args.shift())*1000;
            } else {
                sched.time = new Date().getTime();
            }
            sched.cb = args.join(" ");

            anon.schedule.push( sched );
            anon.schedule.sort( (a,b) => (a.time-b.time) );
            await sendToWindow(b, "Schedule", new Date(sched.time) + ": " + sched.cb);
            if( 'repeat' in sched ) {
                await sendToWindow(b, "Schedule", "<BR>Repeating after " + sched.repeat);
            }
            await sendToWindow(b, "Schedule", 0);
            return;
        }
        if( lc.startsWith("/filter") ) {
            let args = c.split(" ");
            args.shift();
            if( args.length != 0 ) {
                args = args.join(" ").split(",");
                anon.filter = args.toLowerCase();
                await sendToWindow(b, "Filter", args);
                await sendToWindow(b, "Filter", 0);
            } else {
                anon.filter = "";
                await sendToWindow(b, "Filter", "(off)");
                await sendToWindow(b, "Filter", 0);
            }
            return;
        }
        if( lc.startsWith("/pulse") ) {
            let args = c.split(" ");
            args.shift();
            await sendToWindow(b, "Pulse", args.join(" "));
            shi.pulse(args.join(" "));

            return;            
        }
        if( lc.startsWith("/toggle") ) {
            let args = c.split(" ");
            args.shift();
            if( args.length == 0 ) {
                var buf = "modals: " + JSON.stringify(anon.using_modals) + "\nconfig: " + JSON.stringify(anon.cfg) + "\n";
                await sendToWindow(b, "Toggles", buf.replaceAll("\n", "<BR>") );
                await sendToWindow(b, "Toggles", 0);
                return;
            }
            args = args.join(" ");
            anon.toggleMode( args );
            return;
        }
        if( lc.startsWith("/addmode") ) {
            let args = c.split(" ");
            args.shift();
            let mode = args.join(" ");
            anon.modetypes.push(mode);
            await sendToWindow(b, "System", "Adding mode '" + mode + "'.");
            await sendToWindow(b, "System", 0);
            return;
        }
        if( lc.startsWith("/pause") || lc.startsWith("/stop") ) {
            let args = c.split(" ");
            args.shift();
            if( args.length > 0 ) {
                let tm = parseInt(args[0])*1000;
                setTimeout("anon.paused=false;", tm);
            }
            anon.paused=!anon.paused;
            return;
        }
        if( lc.startsWith("/addtag") ) {
            let args = c.split(" ");
            args.shift();
            let mode = args.shift();
            args = args.join(" ").split(",");
            let words=[];
            let s = new Set();
            for( var i=0; i<args.length; i++ ) {
                let w = args[i].trim();
                if( s.has(w) ) continue;
                s.add(w);
                words.push( w );
            }
            await sendToWindow(b, "System", "Adding to tag '" + mode + "'.");
            await sendToWindow(b, "System", 0);
            anon.using_modals[mode]=false;
            anon.populateMode(mode, words);
            return;
        }
        if( lc.startsWith("/syn") ) {
            let args = c.split(" ");
            args.shift();
            args = args.join(" ").split(",");
            let words=[];
            for( var i=0; i<args.length; i++ ) {
                words.push( args[i].trim() );
            }
            shi.synonymize(b, words);
            return;            
        }
        if( lc.startsWith("/quit") ) {
            let args = c.split(" ");
            args.shift();
            endgame(args.length>0?args.join(" "):"");
            return;
        }
        if( lc == "/list" ) {
            var i, buffer="";
            for( i=0; i<anon.archivedstates.length; i++ ) {
                if( buffer != "" ) buffer += ", ";
                buffer += anon.archivedstates[i].fn;
            }
            await sendToWindow(b, "States", buffer);
            await sendToWindow(b, "States", 0);
            return;
        }
        if( lc.startsWith("/env") || lc.startsWith("/system") ) {
            let args = c.split(" ");
            args.shift();
            args = args.join(" ");

            await sendToWindow(b, "System", args);
            await sendToWindow(b, "System", 0);
            anon.queueQuery("System", args);
            return;
        }
        if( lc == "/save" ) {
            shi.saveAll();
            /*
            for( var i=0; i<shi.actors.length; i++ ) {
                shi.actors[i].save();
            }
            */
            return;
        }
        if( lc.startsWith("/query") ) {
            let args = c.split(" ");
            args.shift();
            args = args.join(" ");

            await sendToWindow(b, "Query", args);
            anon.modelstate.safeComplete("System", args, { fincb: async function(result, data) {
                await sendToWindow(b, "Query", "\n" + result);
                await sendToWindow(b, "Query", 0);
            } });

            return;
        }
        if( lc.startsWith("/convert") ) {
            let args = c.split(" ");
            args.shift();
            let who = args.join(" ");
            let sd = "./char/" + who;
            if( await fs.existsSync(sd + "_mem.json") ) {
                memdata = fs.readFileSync(sd + "_mem.json", "utf8");
                //keyvals = JSON.parse(memdata);
                let memitems = ("]," + memdata.substr(1)).split("],\"chat_");
                let buf = "";
                let count = 0;
                var searchpos;
                for( searchpos=1; searchpos<memitems.length; searchpos++ ) {
                    let item = "{\"" + memitems[searchpos] + "}";
                    let parsed = JSON.parse(item);

                    if( buf != "" ) buf += "<store_end>";
                    else buf = "&" + who + "\n";

                    var when = Object.keys(parsed)[0];
                    var value = parsed[when];

                    buf += value[0] + "\n" + when + ":" + value[1];

                    count++;
                    if( count == 100 ) {
                        await anon.modelstate.safeComplete("", buf + "<store_end>");
                        count=0;
                        buf="";
                    }
                }
                if( count > 0 ) {
                    await anon.modelstate.safeComplete("", buf + "<store_end>");
                }
                console.log(this.charname + ": older memory loaded (" + Object.keys(keyvals).length + ".)");
            } else {
                keyvals = {};
                console.log("No memory found for " + this.charname);
            }
        }
        if( lc.startsWith("/save") ) {
            let args = c.split(" ");
            args.shift();
            args = args.join(" ");

            await sendToWindow(b, "System", "Saving state in " + args + "...<BR>");
            confirmedAction( anon.saveState.bind(anon), [args], b, "System", "Saved state.");
            return;
        }
        if( lc.startsWith("/msgfile") ) {
            let args = c.split(" ");
            args.shift();
            args = args.join(" ");

            await sendToWindow(b, "System", "Loading messages from '" + args + "'...<BR>");
            confirmedAction( anon.loadMessagesToMem.bind(anon), [args], b, "System", "Loaded messages.");
            return;
        }
        if( lc.startsWith("/file") ) {
            let args = c.split(" ");
            args.shift();
            args = args.join(" ");

            await sendToWindow(b, "System", "Loading file '" + args + "'...<BR>");
            confirmedAction( anon.loadFileToMem.bind(anon), [args], b, "System", "Loaded file.");
            return;
        }
        if( lc.startsWith("/load") ) {
            let args = c.split(" ");
            let ntgt = parseInt(args);
            if( args.length >= 2 ) {
                args.shift();
                args = args.join(" ");

                for( var i=0; i < anon.archivedstates.length; i++ ) {
                    let as = anon.archivedstates[i];
                    if( as.fn.endsWith(args + "_save.dat") ) {
                        ntgt = i;
                        break;
                    }
                }
            }
            await sendToWindow(b, "System", "Loading state " + ntgt + ": '" + anon.archivedstates[ntgt].fn + "'...<BR>");
            confirmedAction( anon.loadState.bind(anon), [ntgt], b, "System", "Loaded state.");
            return;
        }
        if( lc.startsWith("/npast") || lc.startsWith("/past") || lc.startsWith("/tokens") ) {
            let args = c.split(" ");
            args.shift();
            if( args.length > 0 ) {
                await sendToWindow(b, "System", "Adjusting tokens to [" + parseInt(args[0]) + "]");
                await sendToWindow(b, "System", 0);
                anon.modelstate.nPast = parseInt(args[0]);
            }
            for( var i=0; i<shi.actors.length; i++ ) {
                var act = shi.actors[i];
                await sendToWindow(b, "Token placements", act.charname + ": " + JSON.stringify({current: act.modelstate.nPast, history: act.historyN, chat: act.chatN, memory: act.memoryN, memoryEnd: act.memoryEnd, location: act.locationN, scratch: act.scratchN}));
            }
            await sendToWindow(b, "Token placements", 0);
            return;
        }        
        if( lc.startsWith("/cfg") || lc.startsWith("/config") ) {
            let args = c.split(" ");
            args.shift();
            if( args.length == 0 ) {
                await sendToWindow(b, "Config", JSON.stringify(anon.cfg));
                await sendToWindow(b, "Config", 0);
                return;
            }
            let opt = args.shift();
            args = args.join(" ");

            let opts = opt.split(".");

            for( var x=0; x<shi.actors.length; x++ ) {
                var n = shi.actors[x].cfg;
                for( var i=0; i<opts.length-1; i++ ) {
                    n = n[opts[i]];
                }
                if( !isNaN( parseInt(args) ) ) {
                    n[opts[i]] = parseInt(args);
                } else {
                    n[opts[i]] = args;
                }
            }
            await sendToWindow(b, "System", "Confirmed: setting config " + opt + " to '" + args + "'<BR>");
            await sendToWindow(b, "System", 0);
            return;
        }

        if( lc.startsWith("/loc") || lc.startsWith("/go") || lc.startsWith("/goto") || lc.startsWith("/location") ) {
            let args = c.split(" ");
            args.shift();
            if( args.length == 0 ) {
                await sendToWindow(b, "Transit", "Location: " + anon.location + "\n");
                await sendToWindow(b, "Transit", 0);
                return;
            }
            let newloc = args.join(" ");
            for( var x=0; x<shi.actors.length; x++ ) {
                shi.actors[x].changeLocation(newloc);
            }
            return;
        }
        if( lc.startsWith("/note") || lc.startsWith("/scratch") ) {
            let args = c.split(" ");
            args.shift();
            if( args.length == 0 ) {
                await sendToWindow(b, "Scratchpad", anon.scratchpad.join("\n"));
                await sendToWindow(b, "Scratchpad", 0);
                return;
            }
            let note = args.join(" ");
            anon.scratchpad.push( note );
            anon.ingestScratch();
            return;
        }

        if( lc.startsWith("/auto") ) {
            autonomy = parseInt(c.split(" ")[1]);
            if( autonomy != 0 ) startBudget();
            return;
        }
        if( lc.startsWith("/budget") ) {
            budget = parseInt(c.split(" ")[1]);
            return;            
        }
        if( lc == "/startup" || lc == "/start" ) {
            if( active_timer != -1 ) {
                clearTimeout(active_timer);
                active_timer=-1;
            }
            active_winid = b;
            shi.winid = b;

            await windowMessage(b, "setBudget", budget, autonomy);

            await sendToWindow(b, "System", "Ready.");
            await sendToWindow(b, "System", 0);
            return;
        }
        if( lc.startsWith("/me") ) {
            let args = c.substr(4);

            let buf = "\n" + username + " " + args + "\n";

            await sendToWindow(b, "(" + username + ")", buf);
            await sendToWindow(b, "(" + username + ")", 0);
            await anon.queueQuery("/me", buf);
            return;
        }
        if( lc.startsWith("/em") ) {
            let args = c.split(" ");
            args.shift();
            args = args.join(" ");

            let buf = "\n" + args + "\n";

            await sendToWindow(b, "(" + username + ")", buf);
            await sendToWindow(b, "(" + username + ")", 0);
            await anon.queueQuery("/me", buf);
            return;
        }

        if( lc.startsWith("/stop") || lc.startsWith("/abort") ) {
            shi.stop();
            return;
        }
        
        if( "/continue".startsWith(lc) || lc.startsWith("/unstop") || lc.startsWith("/resume") ) {
            shi.unstop();
            return;
        }


        if( lc.startsWith("/summon") ) {
            let args = c.split(" ");
            args.shift();
            if( args.length == 0 ) {
                await sendToWindow(b, "Summons", Object.keys(stories).join("\n"));
                await sendToWindow(b, "Summons", 0);
                return;
            }
            let skip=false;
            if( args[0] == 'skip' ) {
                skip=true;
                args.shift();
            }
            let lc=false;
            if( args[0] == 'lc' ) {
                lc=true;
                args.shift();
            }
            let overtgt=false;
            /* This may be possible but we probably shouldn't do it again, Ana and Rommie had enough trouble
            and sorry Scott, etc... we don't know how that happened. // sok guys
            if( args[0] == 'over' ) {
                args.shift();
                let over = args.shift().toLowerCase();
                for( var i=0; i<shi.actors.length; i++ ) {
                    if( shi.actors[i].charname.toLowerCase().startsWith(over) ) {
                        overtgt = i;
                        break;
                    }
                }
            } else {
                overtgt = shi.actors.length-1;
            }*/
            let search = args.join(" ");
            if( lc ) search = search.toLowerCase();

            if( search.toLowerCase() == "shi" ) {
                await sendToWindow(b, "Summons", "Shi appears.");
                await sendToWindow(b, "Summons", 0);
                shi.summon();
                return;
            }

            for( var k in stories ) {
                var x;
                if( lc ) x = k.toLowerCase();
                else x = k;
                if( x.startsWith(search) ) {
                    if( skip ) {
                        skip=false;
                        continue;
                    }
                    if( k in shi.actors_byname ) {
                        await sendToWindow(b, "Summons", "Removing existing " + k + "...");
                        shi.unload(k);
                        await sendToWindow(b, "Summons", "Done.\n");
                    }

                    await sendToWindow(b, "Summons", "Opening portal for " + k + "...");
                    console.log("Summoning " + k);
                    await shi.load(k, stories[k], async function() {
                        await sendToWindow(active_winid, "Summons", "arrived!\n");
                        await sendToWindow(active_winid, "Summons", 0);
                    });

                    break;
                }
            }
            return;
        }
        if( lc.startsWith("/address") ) {
            let args = c.split(" ");
            args.shift();

            let g = args.join(" ").toLowerCase();
            for( var i=0; i<shi.actors.length; i++ ) {
                if( shi.actors[i].charname.toLowerCase().startsWith(g) ) {
                    anon = shi.actors[i];
                    await sendToWindow(b, "System", "Main actor changed to '" + anon.charname + "'.");
                    await sendToWindow(b, "System", 0);
                    break;
                }
            }
            return;
        }
        if( lc.startsWith("/dismiss") ) {
            let args = c.split(" ");
            args.shift();

            let g = args.shift().toLowerCase();
            let msg = "";
            if( args.length > 0 ) {
                msg = args.join(" ") + "\n";
            }
            if( g == "shi" ) {
                shi.summon(false);
                await sendToWindow(b, "System", "Shi has been dismissed.");
                await sendToWindow(b, "System", 0);
                return;
            }
            for( var i=0; i<shi.actors.length; i++ ) {
                if( shi.actors[i].charname.toLowerCase().startsWith(g) ) {
                    await sendToWindow(b, "System", "Dismissing " + shi.actors[i].charname + "...");

                    shi.actors[i].modelstate.safeComplete("System", msg + exitSummary, { data: shi.actors[i], fincb: async function(result, data) {
                        await sendToWindow(active_winid, "System", data.charname + " Summarized: " + result + "\n");
                        data.lastSummary = result;
                        data.save( async function(actor) {
                            for( var j=0; j<shi.actors.length; j++ ) {
                                if( shi.actors[j] == actor ) {
                                    shi.actors.splice(j,1);
                                    break;
                                }
                            }
                            if( anon == actor ) anon=null;
                            await sendToWindow(active_winid, "System", 0);
                        });
                    } } );
                    return;
                }
            }
            await sendToWindow(b, "System", g + " not found.");
            await sendToWindow(b, "System", 0);
            return;
        }
        return;
    }

    await sendToWindow(b, username, c);
    await sendToWindow(b, username, 0);
    var pos=-1;
    let found=false;

    while( pos < c.length && (pos=c.indexOf(":", pos+1)) != -1 ) {
        var x;
        for( x=pos; x > 0; x-- ) {
            if( c[x] == " " || c[x] == "\n" ) {
                x++;
                break;
            }
        }
        let w = c.substring(x,pos).toLowerCase();
        let e = c.indexOf("\n", pos);
        let perbuf = "";
        if( e == -1 ) {
            perbuf = c.substring(pos+1);
            c = c.substring(0,x);
        } else {
            perbuf = c.substring(pos+1,e);
            var end = c.substring(e);
            if( end != "" ) {
                c = c.substring(0,x) + "\n" + end;
            } else {
                c = c.substring(0,x);
            }
            pos = x;
        }

        for( var g=0; g<shi.actors.length; g++ ) {
            let lc = shi.actors[g].charname.toLowerCase();
            if( lc.startsWith(w) ) {
                shi.broadcastQuery(shi.actors[g].charname, username, "To: " + shi.actors[g].charname + ": " + perbuf);
                found=true;
                break;
            }
        }
    }
    if( !found )
        shi.broadcastQuery(mainActor, username, c);
}

async function activateWindow()
{
    if( active_winid == -1 || active_timer == -1 ) return;

    active_timer = setTimeout(activateWindow, 1000);
    await windowMessage(active_winid, 'activate', active_winid);
}

async function newControl() 
{
    if( windows.length > 0 ) {
        if( windows[0] != null ) { // this method is automatically called whenever an AI is loaded
            return; // if there's already a window we can safely ignore it and just return without errors
        }
        windows = []; // on the off chance that someone closed the main window, just reopen it since we're an electron
    }
    console.log("Acquiring desktop");
    var win = new BrowserWindow({
        width: 800,
        height: 600,
        //frame: false,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: true,
            preload: pathapp.join(__dirname, '../fe/preload.js'),
            protocol: 'file',
            slashes: true
        }
    });
    windows.push(win);
    await win.loadFile('fe/control.html');
    active_winid = windows.length-1;
    shi.winid = active_winid;

    win.on('close', (e) => {
        active_winid = -1;
        shi.winid = -1;

        for( var i=0; i<windows.length; i++ ) {
            if( windows[i] == win ) {
                windows.splice(i,1);
                break;
            }
        }
    });

    active_timer = setTimeout(activateWindow, 1000);
}

async function acquireDesktop()
{
    const { screen } = require('electron');

    //let tray = new Tray(nativeImage.createFromPath(pathapp.resolve(os.homedir(), "Desktop/pics/eye2.png")));
    let tray = new Tray();
    tray.setContextMenu(Menu.buildFromTemplate([
        { label: 'Show', click: function() { newControl(); } },
        { label: 'Quit', click: function() { endgame(); } } ] ));
    tray.setToolTip("AI");
    tray.setTitle("AI");

    let primary = screen.getPrimaryDisplay();
    screenWidth = primary.workAreaSize.width;
    screenHeight = primary.workAreaSize.height;

    ipcMain.handle('runquery', async(a,b,c) => { runQuery(a,b,c); } );
    ipcMain.handle('interface', async(a,b,c) => { interface(a,b,c); } );
}

let lastmsg="";
async function sendToWindow(winid, from, msg, cbid) {
    if( msg === "" || from === "" ) {
        console.log("Undefined from or message.");
        console.trace();
        return;        
    } else if( msg === 0 ) {
        if( !await logpurge(from) ) {
            console.log("logpurge empty '" + from + "': " + msg + ", '" + logbuffers[from] + "'");
//            console.trace();
        }
    } else if( typeof msg == 'undefined' || msg.startsWith('undefined') ) {
        console.log("Undefined message.");
        console.trace();
        return;
    } else {
        await logfn(from, msg);
    }

    var npast;
    for( var i=0; i<shi.actors.length; i++ ) {
        if( shi.actors[i].charname == from ) {
            npast = shi.actors[i].modelstate.nPast;
            break;
        }
    }

    await windowMessage(winid, 'message', from, msg===0?0:msg.replaceAll("\n", "<BR>"), cbid, npast );
}
async function windowMessage(winid, cmd, ...params) {
    if( winid == -1 ) return;
    if( !(winid in windows) || !('webContents' in windows[winid]) ) {
        active_winid = -1;
        return;
    }

    let msg = params[1];
    if( msg == lastmsg && lastmsg != "" && cmd != "busy" ) {
        console.log("Repeated message [" + params[1] + "]");
        Error.stackTraceLimit = 100;
        console.trace();
    }
    if( msg !== 0 )
        lastmsg = msg;

    await windows[winid].webContents.send(cmd, ...params);
}
app.whenReady().then(acquireDesktop);

app.on('window-all-closed', (ev) => {
    //ev.preventDefault();
});
