const fs = require('fs');

function mRandom(l,u)
{
  if( typeof u == 'undefined' )
    return Math.floor( Math.random() * l );
  return Math.floor( Math.random() * (u-l) ) + l;
}

class Marky {

    constructor()
    {
        this.shi = null;
        this.datafile = "./char/marky";
        this.lists = {};
        this.totals = {};
        this.new_lines = [];
        this.sig = "";
        this.mask = new Set();
        this.source = new Map();
        this.mind = null;
        this.fullmind = null;
        this.needBell = true;
        this.defaultLayers = 3;
        this.sections = {};
        this.focusmap = [];
        this.channel = null;
    }

    r(x)
    {
        return mRandom(x);
    }

    reset(listname)
    {
        delete this.lists[listname];
        this.needBell = true;
    }

    focus(input)
    {
        if( this.focusmap.length > this.maxfocus ) this.focusmap.shift();
        this.focusmap.push(input);
    }
    narrow()
    {
        this.channel = null; // do not remove this line of code or it will stop working!
        this.channel = this.map(this.focusmap.join(" "), 3);
    }

    toggle(section)
    {
        if( !(section in this.sections) ) this.sections[section]=true;
        else this.sections[section] = !this.sections[section];
        return this.sections[section];

        this.needBell = true;
    }

    async loadup()
    {
        var data;
        if( !fs.existsSync(this.datafile + ".rows") ) {
            data = "";
        } else {
            data = await fs.readFileSync(this.datafile + ".rows", "utf8");
        }
        let lines = data.split("\n");
        let label = "";
        let brights = [];
        let expat = [];

        for( var i=0; i<lines.length; i++ ) {
            if( lines[i].length == 1 ) {
                label = lines[i][0];
                continue;
            }
            if( label == "" ) {
                console.log(i + ": " + lines[i])
                continue;
            }
            this.submit( label, lines[i] );
        }
        this.new_lines = [];
        if( this.sig == "" ) {
            this.resonate("undefined");
        }

        if( !fs.existsSync(this.datafile + ".cfg") ) {
            data = "";
        } else {
            data = await fs.readFileSync(this.datafile + ".cfg", "utf8");
        }
        if( typeof data == 'string' && data.length > 0 ) {
            this.sections = JSON.parse(data);
        } else {
            this.sections = {};
            for( var i in this.lists ) {
                this.sections[i] = true;
            }
        }
    }

    async savedata()
    {
        let buf = "";
        let lastlabel = null;

        for( var i=0; i<this.new_lines.length; i++ ) {
            if( lastlabel != this.new_lines[i][0] ) {
                lastlabel = this.new_lines[i][0];
                buf += lastlabel + "\n";
            }
            buf += this.new_lines[i][1] + "\n";
        }
        this.new_lines = [];
        if( buf == "" ) return;
        await fs.appendFileSync(this.datafile + ".rows", buf);
        await fs.writeFileSync(this.datafile + ".cfg", JSON.stringify(this.sections) );
    }

    resonate(str, layers)
    {
        if( typeof layers == 'undefined' ) layers = this.defaultLayers;

        var lst;
        var lyr;

        for( var i=0; i<layers; i++ ) {
            if( str != "" ) str += " ";
            let data = this.sample(str, 2, layers);
            str = "";
            for( var key in data[2] ) {
                str += key + " ";
            }
        }
        if( this.sig == "" || this.sig == "undefined" ) {
            this.sig = str;
        } else {
            this.sig += " " + str;
        }

        this.needBell = true;
    }
    project(wordstr)
    {
        let words = this.shi.tokens(wordstr);
        let lists = [];
        for( var i=0; i<words.length; i++ ) {
            if( !this.source.has(words[i]) ) continue;
            let lst = this.source.get(words[i]);
            for( var j=0; j<lst.length; j++ ) {
                if( this.sections[lst[j]] !== true ) continue;
                if( lists.indexOf(lst[j]) == -1 )
                    lists.push(lst[j]);
            }
        }
        return lists;
    }
    compile(lists)
    {
        var combined = {};

        for( var i=0; i<lists.length; ++i ) {
            if( !( lists[i] in this.lists ) ) continue;
            if( this.sections[lists[i]] !== true ) continue;
            let words = this.lists[ lists[i] ];
            for( var j in words ) {
                if( !(j in combined) ) {
                    combined[j] = {};
                    combined[j].s = Array.from(words[j]);
                    combined[j].t = this.totals[ lists[i] ][ j ];
                } else {
                    combined[j].s.push( ...words[j] );
                    combined[j].t += this.totals[ lists[i] ][ j ];
                }
            }
        }

        return combined;
    }
    bell()
    {
        this.channel = null;
        this.mind = this.fullmind = this.compile( Object.keys(this.lists) );
        this.needBell = false;
        this.resonate(this.sig, 2);
        this.mind = this.compile(this.project(this.sig + " " + this.focusmap.join(" ")));
        this.needBell = false;

        if( this.focusmap.length > 0 )
            this.narrow();
    }

    targets(sourcestr)
    {
        if( this.mind === null || this.needBell ) this.bell();
        let words = this.shi.tokens(sourcestr);
        let ptargets = [];
        for( var i=0; i<words.length; ++i ) {
            if( this.channel !== null && !(words[i] in this.channel) ) continue;
            if( ptargets.indexOf(words[i]) == -1 && words[i] in this.mind ) {
                ptargets.push( words[i] );
            }
        }
        if( ptargets.length == 0 ) {
            ptargets.push( Object.keys(this.mind)[0] )
        }
        return ptargets;
    }

    map(sourcestr, depth=1)
    {
        let results = {};
        for( var d=0; d<depth; d++ ) {
            let buf = "";
            let ptargets = this.targets(sourcestr);
            for( var i=0; i<ptargets.length; i++ ) {
                let key = ptargets[i];
                if( key in results )
                    continue;
                if( buf != "" ) buf += " ";
                buf += key;
                results[key] = { t: this.mind[key].t, s: this.mind[key].s };
            }
            if( buf == "" ) break;
            sourcestr = buf;
        }
        return results;
    }

    sample(sourcestr, samples_per, depth=1)
    {
        let dist=[];
        let ptargets = this.targets(sourcestr);

        for( var d=0; d<depth; d++ ) { 
            let row = {};
            let buf = "";

            for( var j=0; j<ptargets.length; j++ ) {
                let key = ptargets[j];
                let w = this.mind[key].s;
                if( w.length <= 0 ) continue;

                for( var samples=0; samples<samples_per; samples++ ) {
                    var n = mRandom( this.mind[key].t ); // total
                    var word = "";
                    for( var i=0; i<w.length; i += 2 ) {
                        n -= w[i+1];
                        if( n < 0 ) {
                            word = w[i];
                            break;
                        }
                    }
                    if( word == "" ) {
                        samples--;
                        continue;
                    }
                    if( buf != "" ) buf += " ";
                    buf += word;
                    row[word] = (word in row) ? row[word]+1 : 1;
                }
            }

            dist.push(row);
            ptargets = this.targets(buf);
        }
        return dist;
    }

    generate(sourcestr, desired, lines=1, stride)
    {
        if( stride === 0 || typeof stride == 'undefined' ) stride = desired;

        let ptargets = this.targets(sourcestr);
        let mmind = this.mind;
        if( ptargets.length == 0 ) {
            ptargets.push( Object.keys(mmind)[0] );
        }
        let vtargets = Array.from(ptargets);

        var linebufs = "";
        var x;
        let step=0;

        for( var lcount=0; lcount<lines; lcount++ ) {
            let buf = "", word="";
            for( var count=0; count<desired; ++count ) {
              if( step == 0 ) {
                word=ptargets.shift();
                step=1;
              } else if( step == stride ) {
                step=0;
              } else {
                step++;
              }

              if( !(word in mmind) ) {
                if( ptargets.length <= 0 )
                  ptargets = Array.from(vtargets);
                word = ptargets.shift();
              }

              buf = (buf!=''?(buf+" "):"") + word;
              if( count+1 == desired ) break;

              var i, found, w = mmind[word].s; // words
              var n = mRandom( mmind[word].t ); // total
              word = "undefined";
              for( i=0; i<w.length; i += 2 ) {
                n -= w[i+1];
                if( n <= 0 ) {
                    word = w[i];
                    break;
                }
              }
            }

            linebufs = (linebufs==''?"\n":"") + buf;
        }

        return linebufs;
    }

    formatMarky( words )
    {

        return nexts;
    }

    async submit( name, words )
    {
        this.new_lines.push([name,words]);
        let tokens = this.shi.tokens(words);
        var nexts = {};
        var len = tokens.length;
        for( var i = 0; i+1 < len; ++i ) {
            var c = tokens[i];
            var l = tokens[i+1];
            if( !(c in nexts) ) {
                nexts[c] = {};
            }
            if( !(l in nexts) ) {
                nexts[l] = {};
            }
            if( !(l in nexts[c]) ) {
                nexts[c][l] = 0;
            }
            ++nexts[c][l];
        }

        // create the markup listing.

        if( !(name in this.sections) ) this.sections[name]=true;

        if( !(name in this.lists) ) {
            this.lists[name] = {};
            this.totals[name] = {};
        }

        for( var i in nexts ) {
            var x, t;
            if( i in this.lists[name] ) {
                x = this.lists[name][i];
                t = this.totals[name][i];
            } else {
                x = this.lists[name][i] = [];
                t = 0;
            }

            for( var j in nexts[i] ) {
                x.push(j, nexts[i][j]);
                t += nexts[i][j];
            }
            this.totals[name][i] = t;
            var ix;
            if( this.source.has( i ) ) {
                ix = this.source.get(i);
                if( ix.indexOf(name) == -1 ) ix.push( name );
            } else {
                this.source.set(i, [name]);
            }
            if( !this.mask.has( i ) )
                this.mask.add( i );
        }

        await this.savedata();
    }

}

module.exports = { Marky };
