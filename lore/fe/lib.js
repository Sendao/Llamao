// Copyright 2012-2014 Scott Powell, all rights reserved
// Please feel free to modify and use this code for your own purposes.


var rxCmd={};
function logCommandHandler( reqid, handler, data )
{
    rxCmd[reqid] = [ handler, data ];
}
function shellMessage(msg)
{
    var mx = radVar("mainsh");

    if( mx == null ) {
        radStore("mainsh.text", msg + "<BR>");
    } else {
        mx.text = msg + "<BR>" + mx.text;
    }
    radChange("mainsh");
//  radStore("mainsh.text", mx.text);
}
/* new handler */
function reportLog(reqid, loglines)
{
// send updates to reqid registry:
    var cl, i, store_log=true;
    if( reqid != -1 ) {
        if( reqid in rxCmd ) {
            if( rxCmd[reqid][0]( rxCmd[reqid][1], loglines ) === false ) store_log=false;
            delete rxCmd[reqid];
        }
    }
    if( !store_log ) return;

    cl = radVar("cmdlog");
    if( cl == null ) {
        radStore("cmdlog",[]);
        cl = radVar("cmdlog");
    }
    for( i=0; i<loglines.length; i++ ) {
        cl.push( loglines[i] );
        shellMessage(loglines[i]);
    }
    //!verify cmdlog updates properly.
}

/* old handlers: */
var oldlog=[];
var logged_errors=false,logged_messages=false;
function appLog(jcode,msg)
{
    shellMessage( msg );
    logged_messages=true;
    oldlog.push([0,jcode,msg]);
    radStore("oldlog", oldlog);
}
function errorLog(jcode,msg)
{
    shellMessage( "err: " + msg );
    logged_errors=true;
    oldlog.push([1,jcode,msg]);
    radStore("oldlog", oldlog);
}
function finishStatus(jcode)
{
//  oldlog.push([5,jcode]);
//  radStore("oldlog", oldlog);
}

var lib_handlers={};
function handle(cmd,handler,params)
{
    lib_handlers[cmd]=[handler,params];
}

var cmd_handlers={};
function execute(s,cmd,params,handler)
{
    var rid = digi_rand(7);
    var obj = params;
    obj['r']=rid;
    obj['m']=cmd;
    cmd_handlers[rid]=[handler,obj,s,cmd];
    VstPost(s,obj);
}

var cmd_log=[];
function response(rcode,scode,msg,fun)
{
    var fmsg,cmd="",act="";

    if( rcode < 0 )
        fmsg = "<font color=red>" + msg + "</font>";
    else
        fmsg = msg;

    fmsg = linkify(fmsg);

    if( cmd_handlers[rcode] ) {
        cmd = cmd_handlers[rcode][3];
        act = cmd_handlers[rcode][2];
    }

    var obj = { 'code': scode, 'msg': msg, 'text': fmsg, 'qry': rcode, 'act': act, 'cmd': cmd };
    cmd_log.unshift( obj );
    radStore("cmd_log", cmd_log);

    if( cmd_handlers[rcode] ) {
        if( cmd_handlers[rcode][0] )
            cmd_handlers[rcode][0]( cmd_handlers[rcode][1], scode, msg );
        delete(cmd_handlers[rcode]);
    }

    if( lib_handlers['scripts'] )
        lib_handlers['scripts'][0]( obj, lib_handlers['scripts'][1] );

    //!logMessage(fmsg);

    if( typeof fun != 'undefined' ) {
        setTimeout(fun,100);
    }

    delete cmd_handlers[rcode];
}



var logbuf=[];
function alog(str)
{
    logbuf.unshift(str);
    radStore("currentstatus", str);
    radStore("statuslines", logbuf);
}
function logMessage(str)
{
    alog(str);
}

function safeString(str)
{
    if( typeof str == 'undefined' || str == null ) return "";
    return str;
}
//$tx_safe_codes = array("&[am]","&[cm]","&[eq]","&ba","&at");
//$tx_safe_escapes = array("&",",","=","!","@");

var nest_codes = [ "&[am]", "&[cm]", "&[eq]", "&ba", "&at" ];
//"&cm", "&eq", "&am" ];
var nest_escapes = ["&",",","=","!","@"];
//[ ",", "=", "&" ];

function mReplace( str, src, rep )
{
    var i;
    if( typeof str == 'undefined' )
        return "";
    if( typeof str == 'number' )
        return str;
    for(i=0;i<src.length;i++){
        str = str_replace(src[i],rep[i],str);
    }
    return str;
}


function findChild( par, classSearch )
{
    var i;
    var cx = [par.childNodes], kx;

    while( cx.length > 0 ) {
        kx = cx.pop();

        for( i=0; i<kx.length; i++ ) {
            if( kx[i].className.indexOf(classSearch) >= 0 ) {
                return kx[i];
            }
            if( kx[i].childNodes && kx[i].childNodes.length > 0 ) {
                cx.push( kx[i].childNodes );
            }
        }
    }
    return false;
}

function rPrint(y)
{
    var str="";
    for( var i in y ) {
        if( typeof y[i] == 'object' ) {
            str += i + ": obj\n";
        } else {
            str += i + ": " + y[i] + "\n";
        }
    }
    return str;
}

var fpr=[];
function fPrint(y,lev)
{
    var str="";
    if( lev <= 0 ) return "";
    if( typeof y != 'object' ) return y;
    if( typeof y.length != 'undefined' ) {
        for( var i=0; i<y.length; i++ ) {
            if( typeof y[i] == 'object' ) {
                str += i + ":";
                if( y[i] == null ) str += "null\n";
                else str += "\n{" + fPrint(y[i],lev-1) + "}\n\n";
            } else {
                str += i + ": '" + y[i] + "'\n";
            }
        }
    } else {
        var fnd=false;
        for( var i in y ) {
            fnd=true;
            if( typeof y[i] == 'object' ) {
                str += i + ":";
                if( y[i] == null ) str += "null\n";
                else str += "\n{" + fPrint(y[i],lev-1) + "}\n\n";
            } else {
                str += i + ": '" + y[i] + "'\n";
            }
        }
        if( !fnd ) {
            str += "{" + y + "}";
        }
    }
    return str;
}

function createCookie(name,value,days) {
    if (days) {
        var date = new Date();
        date.setTime(date.getTime()+(days*24*60*60*1000));
        var expires = "; expires="+date.toGMTString();
    }
    else var expires = "";
    document.cookie = /* document.cookie + ";" + */ name+"="+value+expires+"; SameSite=strict; path=/";
}

function _readCookie(name) {
  var nameEQ = name + "=";
  var ca = document.cookie.split(";");//cooks[1].split(',');
  for(var i=0;i < ca.length;i++) {
      var c = ca[i];
      while (c.charAt(0)==' ') c = c.substring(1,c.length);
      if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length,c.length);
  }
  return null;
}
function readCookie(name) {
  var j = _readCookie("cookies");
  if( j == null ) {
    return _readCookie(name);
  }
  var cookies = j.split(",");
  for( var i=0; i<cookies.length; i++ ) {
    var cook = cookies[i].split("=");
    if( cook[0] == name ) {
      createCookie( name, cook[1] );
      return cook[1];
    }
  }
  return null;
}

function fixMarkup(xhack)
{
    return xhack.replace(/>/g,'&gt;').replace(/</g,'&lt;').replace(/&/g,'&amp;');
}
//API

function clearNodeFast(v)
{
    while( v.childNodes.length > 0 ) {
        v.removeChild( v.firstChild );
    }
}
function clearNode(v)
{
    var xv=[], xr=[];
    if( v == null || typeof v.childNodes == 'undefined' ) return;

    for(var i=0;i<v.childNodes.length;i++){
        xv.push(v.childNodes[i]);
    }
    while( xv.length > 0 ) {
        v = xv.shift();
        if( v.nodeName == '#comment' || !isValid(v.childNodes) ) {
            v.parentNode.removeChild(v);
            continue;
        }
        for(i=0;i<v.childNodes.length;i++){
            xv.push(v.childNodes[i]);
        }
        xr.push(v);
    }
    while( xr.length > 0 ) {
        v = xr.shift();
        if( isValid(v.xChart) ) {
            radChartClear( v.xChart, v.croot, v.xSeed );
        } else if( isValid(v.xLoop) ) {
            // radLoopClear( v.xLoop, v.xSeed );
        }
        v.parentNode.removeChild(v);
    }
}
function kamiNode(v)
{
    if( !v || !v.parentNode ) return;
    v.parentNode.removeChild(v);
}
function isWhite(s) {
    var c,i;
    for(i=0;i<s.length;i++){
        c=s.substr(i,1);
        if(c==' '||c=='\t'||c=='\n'||c=='\r')continue;
        return false;
    }
    return true;
}
if( typeof isValid == 'undefined' || !isValid ) {
function isValid(v)
{
    if ( typeof v != 'undefined' && v !== false && v !== null && v !== "" ) return v;
    else return false;
}
}
function isValidString(v)
{
    if( typeof v == 'string' ) return v;
    else return "";
}
function isValidOrEmpty(v)
{
    if ( typeof v != 'undefined' && v !== false && v !== null ) return true;
    else return false;
}

var isalpha_ab = "A".charCodeAt(0);
var isalpha_zb = "Z".charCodeAt(0);
//Upper and lower bounds for lower case characters
var isalpha_aa = "a".charCodeAt(0);
var isalpha_za = "z".charCodeAt(0);
function isAlpha(s) {
    var c = s.charCodeAt(0);
    if( ( c >= isalpha_aa && c <= isalpha_za ) || ( c >= isalpha_ab && c <= isalpha_zb ) ) {
        return true;
    }
    return false;
}
function isNumber(n) {
  return !isNaN(parseFloat(n)) && isFinite(n);
}
function ctype_alnum(x)
{
    return ( isAlpha(x) || isNumber(x) );
}
String.prototype.capitalize = function() {
    return this.charAt(0).toUpperCase() + this.slice(1);
}
function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
function cCL()
{
    var cl = cDivCl("cl");
    cl.innerHTML="&nbsp;";
    return cl;
}
function cFlText(txt)
{
    var cl = cDivCl("fl");
    cl.appendChild( cText(txt) );
    return cl;
}
function cFold(el,txt)
{
    var cl = cE(el);
    cl.appendChild( cText(txt) );
    return cl;
}
function aCl(div,cls)
{
    if( div.className != "" ) div.className += " " + cls;
    else div.className=cls;
}
function cClText(col,text)
{
    var v = cDivCl("fl");
    v.style.color = col;
    aC(v, cText(text));
    return v;
}
function cOption(val,disp)
{
    var opt = cE("option");
    opt.value=val;
    if(!isValid(disp))disp=val;
    opt.innerHTML=disp;
    return opt;
}
function cFLT(txt) { return cFlText(txt); }
function cText(text) { return document.createTextNode(text); }
function getAttribute(dv, e) {
	var i, eL = e.toLowerCase();
    if( !dv.attributes ) return null;
	for( i = dv.attributes.length-1; i>=0; --i ) {
		if( dv.attributes[i].name.toLowerCase() == eL ) {
			return dv.attributes[i].value;
		}
	}
	return null;
}
function gE(id) { return document.getElementById(id); }
function cE(elName) { return document.createElement(elName); }
function cC() { return cDivCl("cl"); }
function cBR() { return cDivCl("cl"); }
function sA(a,b,c) { a.setAttribute(b,c); }
function hF(a,b) { var hf = cE('input'); sA(hf,'type','hidden'); sA(hf,'name',a); sA(hf,'value',b); return hf; }
function cWrap(p,c) { p.appendChild(c); return p; }
function cBR() { return cDivCl("cl"); }
function aC(p,c) { p.appendChild(c); }
function aCE(p,olda,newb) { p.insertBefore(newb,olda); }
function imap(a,b) { for (var i in a) b(i,a[i]); }
function cmeyes(d,pad,bg,bd,cl)
{
    d.style.padding = pad + "px";
    d.style.background = bg;
    if( bd )
        d.style.border = '1px solid black';
    if( typeof cl != 'undefined' )
        d.style.color = cl;
}
function cT(txt)
{
    var cl = cDivCl("cl");
    cl.appendChild( cText(txt) );
    cl.appendChild( cDivCl("cl") );
    return cl;
}
function cCheck(n,v)
{
    var cl = cE("input");
    cl.type = 'checkbox';
    cl.name = n;
    cl.value = v;
    return cl;
}
function cDivCl(className)
{
    var cl = cE("div");
    cl.className = className;
    return cl;
}
function cDivId(idName)
{
    var cl = cE("div");
    cl.id = idName;
    return cl;
}

function cDivText(txt)
{
    var cl = cE("div");
    cl.appendChild( cText(txt) );
    return cl;
}
function cHidden(elName,elValue)
{
    var el = cE("input");
    el.type = "hidden";
    el.name = elName;
    el.value = elValue;
    return el;
}
function cUpload(elName)
{
    var el = cE("input");
    el.type = 'file';
    el.name = elName;
    return el;
}
function cTextbox(elName,elValue)
{
    var el = cE("input");
    el.type = "text";
    el.size = 60;
    el.name = elName;
    el.value = elValue;
    return el;
}
function cSubmit(elLabel)
{
    var el = cE("input");
    el.type = "submit";
    el.value = elLabel;
    return el;
}
function cTextarea(elName,elValue)
{
    var el = cE("textarea");
//  el.style.width = '200px';
//  el.style.height = '60px';
    el.name = elName;
    el.innerHTML = elValue;
    return el;
}
function cBtn0(txt,clk)
{
    var btn = cDivCl("btn0");
    btn.setAttribute("onclick", clk);
    btn.innerHTML = txt;
    return btn;
}
function cBtnD(txt,clk)
{
    var btn = cDivCl("cli");
    btn.setAttribute("onclick", clk);
    btn.innerHTML = txt;
    return btn;
}
function mp(e)
{
    if (e.pageX || e.pageY)     {
        posx = e.pageX;
        posy = e.pageY;
    }
    else if (e.clientX || e.clientY)    {
        posx = e.clientX + document.body.scrollLeft
            + document.documentElement.scrollLeft;
        posy = e.clientY + document.body.scrollTop
            + document.documentElement.scrollTop;
    }
    return [ posx, posy ];
}
function cBtn(txt,clk)
{
    var cl = cDivCl("fl");
    cl.style.cursor='pointer';
    cl.setAttribute('onclick',clk);
    cl.innerHTML=txt;
    return cl;
}
function cSBtn(txt,clk)
{
    var cl = cDivCl("fl");
    cl.style.cursor='pointer';
    cl.setAttribute('onmouseup',clk);
    cl.innerHTML=txt;
    return cl;
}
function cPad(el,m,l,r,t,b)
{
    if( m == "" || m == 0 || m === false ) {
        el.style.paddingLeft = l + "px";
        el.style.paddingRight = r + "px";
        el.style.paddingBottom = b + "px";
        el.style.paddingTop = t + "px";
    } else {
        el.style.paddingLeft = m + "px";
        el.style.paddingRight = m + "px";
        el.style.paddingBottom = m + "px";
        el.style.paddingTop = m + "px";
    }
    return el;
}
function autofform( url, parms, target )
{
    var afparms = parms.split(/&/);
    var vparms={},vxparms;
    for( var i in afparms ) {
        vxparms = afparms[i].split(/=/);
        if( vxparms.length == 1 )
            vparms[vxparms[0]] = "";
        else
            vparms[vxparms[0]] = vxparms[1];
    }
    return autoform(url,vparms,target);
}
function autoform( url, parms, target )
{
    var frm=cE("form");
    frm.method = 'POST';
    frm.action = url;
    var cH;
    for( var i in parms ) {
        aC(frm,cHidden( i, parms[i] ));
    }
    if( typeof target != 'undefined' ) frm.target = target;
    frm.submit();
}
function waitForVis(divid,cmd)
{
    if( !isValid(divid) ) return -1;
    var div = gE(divid);
    if( div == null ) return -1;

    if( div.style.display == 'none' ) {
        setTimeout('waitForVis("'+divid+'","'+cmd+'")',100);
    } else {
        eval(cmd);
    }
}
function isAncestorOf( child, parent )
{
    var el = child;
    while( el ) {
        if( el == parent ) return true;
        el = el.parentNode;
    }
    return false;
}
function autoSZ(mm,topage,xcenter,ycenter,xscale,xreduce,yscale,yreduce)
{
    if( mm.style.display == 'none' ) {
        dbg("no display");
        return;
    }
    if( !isValid(xscale) ) xscale=0;
    if( !isValid(xreduce) ) xreduce=0;
    if( !isValid(yscale) ) yscale=0;
    if( !isValid(yreduce) ) yreduce=0;
    var gr,xp,yp;
    var tw = false, th = false;
    var ww = 0, hh = 0;
    if( topage!=1 && mm.parentNode ) {
        ww = mm.parentNode.clientWidth;
        hh = mm.parentNode.clientHeight;
    }
    if( isNaN(ww) || ww == 0 ) ww = winW;
    if( isNaN(hh) || hh == 0 ) hh = winH;

    gr = parseInt(getStyle(mm, 'margin-left'));
    if( isNaN(gr) ) gr=0;
    xp=gr;
    gr = parseInt(getStyle(mm, 'margin-right'));
    if( isNaN(gr) ) gr=0;
    xp+=gr;

    gr = parseInt(getStyle(mm, 'margin-top'));
    if( isNaN(gr) ) gr=0;
    yp=gr;
    gr = parseInt(getStyle(mm, 'margin-bottom'));
    if( isNaN(gr) ) gr=0;
    yp+=gr;

//  if( xcenter != 1 )
        if( xp != 0 )
            ww -= xp;
    //if( xp != 0 ) ww -= xp;
    if( xscale != 0 ) tw = ww*xscale;
    if( xreduce != 0 ) {
        if( xscale == 0 ) tw = (xreduce>0)?(ww-xreduce):(-xreduce);
        else tw -= xreduce;
    }

//  if( ycenter != 1 )
        if( yp != 0 ) hh -= yp;
    if( yscale != 0 ) th = hh*yscale;
    if( yreduce != 0 ) {
        if( yscale == 0 ) th = (yreduce>0)?(hh-yreduce):(-yreduce);
        else th -= yreduce;
    }


    if( tw !== false ) {
        if( mm.nodeName == 'IMG' ) {
            mm.width = tw;
        } else {
            mm.style.maxWidth = mm.style.width = tw + "px";
            mm.clientWidth = tw;
        }
    }
    if( xcenter == 1 )
        if( topage == 1 )
            mm.style.left = (ww-tw)/2 + "px";
        else
            mm.style.marginLeft = (ww-tw)/2 + "px";

    if( th !== false ) {
        if( mm.nodeName == 'IMG' ) {
            mm.height = th;
        } else {
            mm.style.maxHeight = mm.style.height = th + "px";
        }
        mm.clientHeight = th;
    }
    if( ycenter == 1 )
        if( topage == 1 )
            mm.style.top = (hh-th)/2 + "px";
        else
            mm.style.marginTop = (hh-th)/2 + "px";

}

var mdivs=[];
function autoSM(mm,xs,xr,ys,yr,xa,ya)
{
    if( mm.style.display == 'none' ) {
        dbg("no display");
        return;
    }
    var i;
    for(i=0;i<mdivs.length;i+=2) {
        if( mdivs[i].id == mm.id )
            break;
    }
    mdivs[i] = mm;
    mdivs[i+1]=[xs,xr,ys,yr,xa,ya];

    XRegisterResize('autoMSize2("' + mm.id + '")');
//  if( mm.style.display != 'none' ) {
        setTimeout('autoMSize2("' + mm.id + '")',1);
//  } else {
//      waitForVis(mm.id,'autoMSize2("' + mm.id + '")');
//  }
}
function autoMSize2(divid)
{
    if( !isValid(divid) ) { console.error("missing div id"); return -1; }
    var i;
    for(i=0;i<mdivs.length;i+=2) {
        if( mdivs[i].id == divid )
            break;
    }
    if( i >= mdivs.length ) { console.error("can't locate " + divid); return; }

    var mm=gE(divid);
    if( !mm ) {
        mdivs.splice(i,2);
        return -1;
    }
    var xy = getPos( mm );
    var v=mdivs[i+1];
    var xs=v[0],xr=v[1],ys=v[2],yr=v[3],xa=v[4],ya=v[5];

//  mm.style.visibility = 'visible';
    var ww = winW;//mm.parentNode.clientWidth;
    var hh = winH;//mm.parentNode.clientHeight;
    var th = false, tw = false;

    var qr;

    var xp=0;
    qr = parseInt(getStyle(mm, 'padding-left'));
    if( isNaN(qr) ) qr=0;
    xp=qr;
    qr = parseInt(getStyle(mm, 'padding-right'));
    if( isNaN(qr) ) qr=0;
    xp+=qr;
/*
    qr = parseInt(getStyle(mm, 'margin-left'));
    if( isNaN(qr) ) qr=0;
    xp+=qr;
    qr = parseInt(getStyle(mm, 'margin-right'));
    if( isNaN(qr) ) qr=0;
    xp+=qr;
*/
    var yp=0;
    qr = parseInt(getStyle(mm, 'padding-top'));
    if( isNaN(qr) ) qr=0;
    yp=qr;
    qr = parseInt(getStyle(mm, 'padding-bottom'));
    if( isNaN(qr) ) qr=0;
    yp+=qr;
    /*
    qr = parseInt(getStyle(mm, 'margin-top'));
    if( isNaN(qr) ) qr=0;
    yp+=qr;
    qr = parseInt(getStyle(mm, 'margin-bottom'));
    if( isNaN(qr) ) qr=0;
    yp+=qr;
*/
    if( xp != 0 ) ww -= xp;
    if( xs != 0 ) tw = ww*xs;
    if( xr != 0 ) {
        if( xs == 0 ) tw = (xr>0)?(ww-xr):(-xr);
        else tw -= xr;
    }
    if( tw !== false && tw >= 0 ) {
        mm.style.maxWidth = mm.style.width = tw + "px";
        //mm.clientWidth = tw;
    }

    if( yp != 0 ) hh -= yp;
    if( ys != 0 ) th = hh*ys;
    if( yr != 0 ) {
        if( ys == 0 ) th = (yr>0)?(hh-yr):(-yr);
        else th -= yr;
    }
    if( th !== false && th >= 0 ) {
//      console.log(th + ": " + ys + ", " + hh + ", " + yp + "(" + winH + ")");
        mm.style.maxHeight = mm.style.height = th + "px";
        //mm.clientHeight = th;
    }
}

var omdivs=[];
function omv(mm,loc,xs,xr,ys,yr)
{
    if( mm.style.display == 'none' ) {
        dbg("no display");
        return;
    }
    var i;
    for(i=0;i<omdivs.length;i+=2) {
        if( omdivs[i].id == mm.id )
            break;
    }
    omdivs[i] = mm;
    omdivs[i+1]=[xs,xr,ys,yr,loc];

    XRegisterResize('autoOMV2("' + mm.id + '")');
//  if( mm.style.display != 'none' ) {
        setTimeout('autoOMV2("' + mm.id + '")',1);
//  } else {
//      waitForVis(mm.id,'autoMSize2("' + mm.id + '")');
//  }
}
function autoOMV2(divid)
{
    if( !isValid(divid) ) { console.error("missing div id"); return -1; }
    var i;
    for(i=0;i<omdivs.length;i+=2) {
        if( omdivs[i].id == divid )
            break;
    }
    if( i >= omdivs.length ) { return -1; }

    var mm=gE(divid);
    if( !mm ) {
        omdivs.splice(i,2);
        return -1;
    }
    var xy = getPos( mm );
    var v=omdivs[i+1];
    var xs=v[0],xr=v[1],ys=v[2],yr=v[3],locs=v[4];
    var xa=false,ya=false;
    var is_global=true;
    var ww = winW;//mm.parentNode.clientWidth;
    var hh = winH;//mm.parentNode.clientHeight;
    var th = false, tw = false;
    var userel=false,useabs=false,usenul=true;
    var adj=[0,0];

    var pxn=mm.parentNode;
    var smloc, smsym;
    for(i=0;i<locs.length;i++){
        switch(locs[i]){
            case 'page':
                pxn = document.documentElement;
                useabs=true; usenul=false;
                break;
            case 'box': case 'parent':
                pxn = mm.parentNode;
                useabs=true; usenul=false;
                break;
            case 'left': xa=-1; break;
            case 'right': xa=1; break;
            case 'center': xa=0; break;
            case 'top': ya=-1; break;
            case 'bot': case 'bottom': ya=1; break;
            case 'middle': ya=0; break;
            case 'rel': case 'relative': userel=true; break;
            case '!rel': case '!relative': userel=false; break;
            case 'abs': case 'absolute': useabs=true; break;
            case '!abs': case '!absolute': useabs=false; break;
            case 'nul': case 'inline': usenul=true; break;
            default:
                smloc = locs[i].substr(0,1);
                smsym = locs[i].substr(1,1);
                if( smloc == 'x' ) {
                    if( smsym == '+' ) {
                        adj[0] += parseInt( locs[i].substr(2) );
                    } else if( smsym == '-' ) {
                        adj[0] -= parseInt( locs[i].substr(2) );
                    }
                } else if( smloc == 'y' ) {
                    if( smsym == '+' ) {
                        adj[1] += parseInt( locs[i].substr(2) );
                    } else if( smsym == '-' ) {
                        adj[1] -= parseInt( locs[i].substr(2) );
                    }
                }
                break;
        }
    }
    var qr;
    var xp=0,yp=0;
    var xm=0,ym=0;
    var xm0=0,ym0=0;

    qr = parseInt(getStyle(mm, 'padding-left'));
    if( isNaN(qr) ) qr=0;
    xp=qr;
    qr = parseInt(getStyle(mm, 'padding-right'));
    if( isNaN(qr) ) qr=0;
    xp+=qr;
    qr = parseInt(getStyle(mm, 'margin-left'));
    if( isNaN(qr) ) qr=0;
    xm+=qr;
    xm0 = qr;
    qr = parseInt(getStyle(mm, 'margin-right'));
    if( isNaN(qr) ) qr=0;
    xm+=qr;


    qr = parseInt(getStyle(mm, 'border-left'));
    if( isNaN(qr) ) qr=0;
    xp+=qr;
    qr = parseInt(getStyle(mm, 'border-right'));
    if( isNaN(qr) ) qr=0;
    xp+=qr;

    qr = parseInt(getStyle(mm, 'padding-top'));
    if( isNaN(qr) ) qr=0;
    yp=qr;
    qr = parseInt(getStyle(mm, 'padding-bottom'));
    if( isNaN(qr) ) qr=0;
    yp+=qr;
    qr = parseInt(getStyle(mm, 'margin-top'));
    if( isNaN(qr) ) qr=0;
    ym+=qr;
    ym0 = qr;
    qr = parseInt(getStyle(mm, 'margin-bottom'));
    if( isNaN(qr) ) qr=0;
    ym+=qr;

    qr = parseInt(getStyle(mm, 'border-top'));
    if( isNaN(qr) ) qr=0;
    yp+=qr;
    qr = parseInt(getStyle(mm, 'border-bottom'));
    if( isNaN(qr) ) qr=0;
    yp+=qr;
//  if( xp != 0 ) ww -= xp;
//  if( xm != 0 ) ww -= xm;

    if( pxn != document.documentElement ) {
        ww = pxn.clientWidth;
        hh = pxn.clientHeight;
    }
    if( !isNaN(xs) && xs != 0 ) tw = ww*xs;
    if( !isNaN(xr) && xr != 0 ) {
        if( xs != 0 ) tw -= xr;
        else if( xr > 0 ) tw = ww-xr;
        else tw = -xr;
    }
    if( tw !== false && tw >= 0 ) {
        tw -= (xm+xp);
        mm.style.maxWidth = mm.style.width = tw + "px";
        //mm.clientWidth = tw;
    }

//  if( yp != 0 ) hh -= yp;
//  if( ym != 0 ) hh -= ym;
    if( ys != 0 ) th = hh*ys;
    if( yr != 0 ) {
        if( ys != 0 ) th -= yr;
        else if( yr > 0 ) th = hh-yr;
        else th = -yr;
    }
    if( th !== false && th >= 0 ) {
        th -= (ym+yp);
        mm.style.maxHeight = mm.style.height = th + "px";
        //mm.clientHeight = th;
    }
    if( xa !== false || ya !== false ) {
        if( tw === false ) tw = mm.clientWidth;
        if( th === false ) th = mm.clientHeight;
        var n;
        if( useabs )
            mm.style.position = 'absolute';
        else if( userel )
            mm.style.position = 'relative';

        if( xa === false ) {
            n = xy[0];
        } else if( xa < 0 ) {
            n = 0;
        } else if( xa == 0 ) {
            n = (ww-(tw+xm+xp))/2;
        } else {
            n = ww-(tw+xm+xp);
        }
        if( !isNaN(n) )
            mm.style.left = (n+adj[0]) + "px";

        if( ya === false ) {
            n = xy[1];
        } else if( ya < 0 ) {
            n = 0;
        } else if( ya == 0 ) {
            n = (hh-(th+ym+yp))/2;
        } else {
            n = hh-(th+ym+yp);
        }
        if( !isNaN(n) )
            mm.style.top = (n+adj[1]) + "px";
    }
    return 0;
}
/*
function autoMSize2(divid)
{
    var mm=gE(divid);
    if( !mm ) { console.error("can't find " + divid); return; }
    var xy = getPos( mm );
    var i;
    for(i=0;i<mdivs.length;i+=2) {
        if( mdivs[i].id == divid )
            break;
    }
    if( i >= mdivs.length ) { dbg("autoMSize2: can't locate " + divid); return; }

    var v=mdivs[i+1];
    var xs=v[0],xr=v[1],ys=v[2],yr=v[3];

//  mm.style.visibility = 'visible';
    var ww = winW;//mm.parentNode.clientWidth;
    var hh = winH;//mm.parentNode.clientHeight;
    var th = false, tw = false;

    var qr;

    var xp=0;
    if( isNaN( qr = parseInt(mm.style.paddingLeft) ) ) qr = 0;
    xp=qr;
    if( isNaN( qr = parseInt(mm.style.paddingRight) ) ) qr = 0;
    xp+=qr;
    if( isNaN( qr = parseInt(mm.style.marginLeft) ) ) qr = 0;
    xp+=qr;
    if( isNaN( qr = parseInt(mm.style.marginRight) ) ) qr = 0;
    xp+=qr;

    var yp=0;
    if( isNaN( qr = parseInt(mm.style.paddingTop) ) ) qr = 0;
    yp=qr;
    if( isNaN( qr = parseInt(mm.style.paddingBottom) ) ) qr = 0;
    yp+=qr;
    if( isNaN( qr = parseInt(mm.style.marginTop) ) ) qr = 0;
    yp+=qr;
    if( isNaN( qr = parseInt(mm.style.marginBottom) ) ) qr = 0;
    yp+=qr;

    if( xp != 0 ) ww -= xp;
    if( xs != 0 ) tw = ww*xs;
    if( xr != 0 ) tw = (xs==0)? ( ww-(2*xr) ) : ( tw-(2*xr) );

    if( tw !== false ) {
        mm.style.maxWidth = mm.style.width = tw + "px";
        mm.clientWidth = tw;
    }

    if( yp != 0 ) hh -= yp;
    if( ys != 0 ) th = hh*ys;
    if( yr != 0 ) th = (ys==0)? ( hh-(x*yr) ) : ( th-(2*yr) );

    if( th !== false ) {
        mm.style.maxHeight = mm.style.height = th + "px";
        mm.clientHeight = th;
    }
}
*/
var scaledivs=[];
function autoSx(mm,xs,xp,ys,yp)
{
    var i;
    for(i=0;i<scaledivs.length;i+=2) {
        if( scaledivs[i].id == mm.id )
            break;
    }
    scaledivs[i] = mm;
    scaledivs[i+1]=[xs,xp,typeof ys=='undefined'?0:parseFloat(ys),typeof yp=='undefined'?0:parseInt(yp)];

    XRegisterResize('autoSize2("' + mm.id + '")');
    if( mm.style.display != 'none' ) {
        setTimeout('autoSize2("' + mm.id + '")',1);
    } else {
        waitForVis(mm.id,'autoSize2("' + mm.id + '")');
    }
}
function autoSize2(divid)
{
    var mm=gE(divid);
    if( !mm ) { console.error("can't find " + divid); return; }
    var xy = getPos( mm );
    var i;
    for(i=0;i<scaledivs.length;i+=2) {
        if( scaledivs[i].id == divid )
            break;
    }
    if( i >= scaledivs.length ) {
        dbg("autoSize2: can't locate " + divid); throw "Can't find div"; return;
    }

    var v=scaledivs[i+1];
    var xs=v[0],xp=v[1],ys=v[2],yp=v[3];

//  mm.style.visibility = 'visible';
    var ww = winW;//mm.parentNode.clientWidth;
    var hh = winH;//mm.parentNode.clientHeight;

    if( xp != 0 ) {
        mm.style.marginLeft = ww*xp + "px";
        ww -= ww*(xp*2);
    }
    if( xs != 0 ) {
        mm.style.maxWidth = mm.style.width = (ww * xs) + "px";
        mm.clientWidth = (ww * xs);
    }
    if( yp != 0 ) {
        mm.style.marginBottom = mm.style.marginTop = hh*yp + "px";
        hh -= hh*(yp*2);
    }
    if( ys != 0 ) {
        mm.style.maxHeight = mm.style.height = (hh - ys) + "px";
        mm.clientHeight = (hh - ys);
    }
}
function classSets(div,srch,repl)
{
	var newClass, wasFound = false;
    if( !isValid(div) ) return;
    if( isValid(div.nodeName) && div.nodeName.indexOf(srch)!=-1 ) {
    	wasFound = true;
		newClass = div.nodeName.replace(srch,repl);
		div.nodeName = newClass;
	}
    if( typeof div.attributes != 'undefined' ) {
	    for( var i = 0; i < div.attributes.length; ++i ) {
	    	if( div.attributes[i].value == '' ) {
	    		if( div.attributes[i].name.indexOf(srch) != -1 ) {
	    			wasFound = true;
	    			newClass = div.attributes[i].name.replace(srch,repl);
	    			div.removeAttribute( div.attributes[i].name );
	    			i=0;
	    			div.setAttribute( newClass, '' );
	    		}
	    	}
	    }
    }
    if( isValid(div.className) ) {
    	if( div.className.indexOf(srch)!=-1 ) {
    		wasFound = true;
    		newClass = div.className;
    		while( newClass.indexOf(srch) != -1 ) {
    			newClass = newClass.replace(srch,repl);
    		}
	        div.className = newClass;
	    } else if( !wasFound && div.className.indexOf(repl) == -1 ) {
            div.className += " " + repl;
            wasFound = true;
	    }
    } else if( !wasFound ) {
        div.className = repl;
    }
}
/*
function cloneObject(source) {
    for (i in source) {
        if (typeof source[i] == 'source')
            this[i] = new cloneObject(source[i]);
        else
            this[i] = source[i];
    }
}
*/

function getSize(e)
{
    var wh = [0,0];

    if( e.clientWidth != undefined && !isNaN(e.clientWidth) )
        wh[0] = e.clientWidth;
    else if( e.style.width != "" )
        wh[0] = parseInt(e.style.width);
    if( e.clientHeight != undefined && !isNaN(e.clientHeight) )
        wh[1] = e.clientHeight;
    else if( e.style.height != "" )
        wh[1] = parseInt(e.style.height);
    return wh;
}
function getInnerSize(e)
{
    var br = [0,0];
    var i, x;

    for( i = 0; i < e.childNodes.length; i++ ) {
        x = e.childNodes[i];
        if( x.clientWidth + x.offsetLeft > br[0] ) br[0] = x.clientWidth + x.offsetLeft;
        if( x.clientHeight + x.offsetTop > br[1] ) br[1] = x.clientHeight + x.offsetTop;
    }
    return br;
}

function getPos(e)
{
    var cl = 0, ct=0;
    do
    {
        if( typeof e.offsetLeft == 'undefined' ) break;
        cl += e.offsetLeft;
        ct += e.offsetTop;
    } while ( (e=e.offsetParent) );
    return [cl,ct];
}

function getPosInside(e,inel)
{
    var cl = 0, ct=0;
    do
    {
        if( e == inel ) break;
        if( typeof e.offsetLeft == 'undefined' ) break;
        cl += e.offsetLeft;
        ct += e.offsetTop;
    } while ( (e=e.offsetParent) );
    return [cl,ct];
}

function getScrolls(e)
{
    var cl = 0, ct=0;
    do
    {
        if( typeof e.scrollLeft == 'undefined' ) break;
        cl += e.scrollLeft;
        ct += e.scrollTop;
//      console.log(e.clientTop + " - " + e.scrollTop);
    } while ( e=e.parentNode );
    return [cl,ct];
}

function getPosB(e)
{
    var cl = 0, ct=0;
    do
    {
        if( typeof e.clientLeft == 'undefined' ) break;
        cl += e.clientLeft - e.scrollLeft;
        ct += e.clientTop - e.scrollTop;
//      console.log(e.clientTop + " - " + e.scrollTop);
    } while ( e=e.parentNode );
    return [cl,ct];
}









function dOpen( id )
{
    var dvT = gE(id);
    dvT.style.display='block';
}
function dClose( id )
{
    var dvT = gE(id);
    dvT.style.display='none';
}
function openDiv( dvSelf, openId )
{
    var dvTarget = gE(openId);
    dvTarget.style.display='block';
    dvSelf.setAttribute('onclick', 'closeDiv(this,"'+openId+'")');
}
function closeDiv( dvSelf, openId )
{
    var dvTarget = gE(openId);
    dvTarget.style.display='none';
    dvSelf.setAttribute('onclick', 'openDiv(this,"'+openId+'")');
}



function mouseOverCom(comid)
{
    var rb = gE('delcom' + comid);
    if( !rb ) return;
    rb.style.visibility='visible';
}
function mouseOutCom(comid)
{
    var rb = gE('delcom' + comid);
    if( !rb ) return;
    rb.style.visibility='hidden';
}
function mouseOver(updid)
{
    var rb = gE('del' + updid);
    if( !rb ) return;
    rb.style.visibility='visible';
}
function mouseOut(updid)
{
    var rb = gE('del' + updid);
    if( !rb ) return;
    rb.style.visibility='hidden';
}
// dynamic menu code
var mousehidetimer = -1;
var mouse_over = -1;
function expandMenu(el)
{
    var menuid = el.id.substr(5);
    var v = gE('dexp' + menuid);
    if( mouse_over != -1 ) {
        var ex = gE(mouse_over);
        ex.style.display='none';
    }
    if( window.event != undefined ) window.event.stopPropagation();
    var kp = getPos(el);
    var kil = kp[0];
    v.style.display='block';
    v.style.left = kil + "px";
//  alert("kx: " + ", kj: " + kil);
    mouse_over = 'dexp' + menuid;

//  if( mousehidetimer != -1 ) {
//      clearTimeout(mousehidetimer);
//  }
//  mousehidetimer = setTimeout("hideMenu();",1500);
}
function mouseVis(v)
{
    var tg = v.target ? v.target : v.srcElement;
    var rtg = v.relatedTarget ? v.relatedTarget : v.fromElement;
//  alert( rtg.id );
    if( !tg || !tg.id ) return;
//  var ex = tg.id.substring(0,4);
//  if( ex != "dexp" ) return;

    if( mousehidetimer != -1 ) {
        clearTimeout(mousehidetimer);
    }

    mouse_over = tg.id;
    return;
}
function mouseHid(v)
{
    //avoid 'mouseout' statements from child nodes:
    if( !v ) v = event;
    var rtg = v.relatedTarget ? v.relatedTarget : v.toElement;
    var tg = v.target ? v.target : v.srcElement;
    if( !tg || !tg.id ) return;
    while( rtg && rtg != tg ) {
        rtg = rtg.parentNode;
    }
    if( rtg == tg ) return;
    if( tg.id != mouse_over ) {
        v.cancelBubble=true;
        if( v.stopPropagation) v.stopPropagation();
        return;
    }

    if( mousehidetimer != -1 ) {
        clearTimeout(mousehidetimer);
    }
    mousehidetimer = setTimeout("hideMenu();",500);
    return;
}

function hideMenu()
{
    if( mousehidetimer != -1 ) clearTimeout(mousehidetimer);
    mousehidetimer=-1;
    if( mouse_over == -1 ) return;

    var v = gE(mouse_over);
    v.style.display='none';
    mouse_over=-1;
}

function moveNodes(from,to)
{
    while( from.childNodes.length > 0 ) {
        to.appendChild(from.childNodes[0]);
        from.removeChild(from.childNodes[0]);
    }
}


var open_divs={};

function popdiv(ev)
{
    var realid=ev.id.substring(3)
    var dv = gE(realid);
    if( (dv.style.display == 'none') ||
        (isValid(dv.className)&&dv.className.indexOf("apage")!=-1) ) {
        open_divs[realid]=true;
//      dv.setAttribute('xdisp','block');
        classSwap(dv, 'apage', 'selpage');
    } else {
        open_divs[realid]=false;
        if( !classSwap(dv, 'selpage', 'apage') )
            dv.style.display='none';
//      dv.setAttribute('xdisp','none');
    }
}

var cpops=[];
function popset(st,ev,mod)
{
    var realid = ev.id.substring(3);
    var poptab;// = realid;
    var dv;
    var cp;
    if( st in cpops ) {
        cp=cpops[st];
    } else {
        cp=false;
    }
    if( cp != false && cp != ev.id ) {
        poptab = cp.substring(3);
        dv=gE(poptab);
        if( isValid(x=dv.getAttribute('onclose')) )
            eval(x + "(dv)");
        open_divs[poptab]=false;
        classSwap(dv, 'selpage','apage');
//      dv.setAttribute('xdisp','none');
        var ex=gE(cp);
        classSwap(ex, 'seltab','atab');
    }
    dv = gE(realid);
    if( dv.style.display == 'block' ) {
        if( isValid(x=dv.getAttribute('onclose')) )
            eval(x + "(dv)");
        classSwap(dv, 'selpage','apage');
//      dv.setAttribute('xdisp','none');
        open_divs[realid]=false;
        if( mod == 1 ) {
            ev.innerHTML = " > ";
        } else if( mod == 2 ) {
            classSwap(ev, 'seltab','atab');
        }
        cpops[st]=false;
    } else {
        if( isValid(x=dv.getAttribute('onopen')) )
            eval(x + "(dv)");
        classSwap(dv, 'apage','selpage');
//      dv.setAttribute('xdisp','block');
        open_divs[realid]=true;
        if( mod == 1 ) {
            ev.innerHTML = " V ";
        } else if( mod == 2 ) {
            classSwap(ev, 'atab','seltab');
        }
        cpops[st]=ev.id;
    }
}

function dirpop(st,tgt)
{
    var poptab,realid = tgt;
    var dv,ov,cp;

    if( st in cpops ) {
        cp=cpops[st];
    } else {
        cp=false;
    }
    dv = gE(realid);
    if( cp != false && cp == realid ) return;

    if( open_divs[realid] ) {
        if( isValid(x=dv.getAttribute('onclose')) )
            eval(x + "(dv)");
        dv.style.display='none';
//      classSwap(dv, 'selpage','apage');
        open_divs[realid]=false;
        cpops[st]=false;
    } else {
        if( cp != false ) {
            poptab = cp;
            ov=gE(poptab);
            if( isValid(x=ov.getAttribute('onclose')) )
                eval(x + "(dv)");
            open_divs[poptab]=false;
            ov.style.display='none';
//          classSwap(dv, 'selpage','apage');
        }
        if( isValid(x=dv.getAttribute('onopen')) )
            eval(x + "(dv)");
        dv.style.display='block';
        if( typeof radSchedLoad == 'function' ) radSchedLoad(dv);
//      classSwap(dv, 'apage','selpage');
        open_divs[realid]=true;
        cpops[st]=tgt;
    }
}



var vpops=[];
function simpop(st)
{
    var ev = window.event.target.parentNode;
//  var realid = ev.id.substring(3);
    litpop(st,ev.id);
}
function litpop(st, tagid)
{
    var ev = gE(tagid);
    var realid = tagid.substring(3);
    var poptab;// = realid;
    var dv, cp;

    cp = ( st in vpops ) ? vpops[st] : false;

    if( cp != false ) {
        if( cp == tagid ) return;
        poptab = cp.substring(3);
        dv=gE(poptab);
        if( isValid(x=dv.getAttribute('onclose')) )
            eval(x + "(dv)");
        open_divs[poptab]=false;
        classSwap(dv, 'selpage','apage');
        var ex=gE(cp);
        classSwap(ex, 'seltab','atab');
    }
    dv = gE(realid);
    if( dv.style.display == 'block' ) {
        if( isValid(x=dv.getAttribute('onclose')) )
            eval(x + "(dv)");
        classSwap(dv, 'selpage','apage');
        open_divs[realid]=false;
        classSwap(ev, 'seltab','atab');
        vpops[st]=false;
    } else {
        if( isValid(x=dv.getAttribute('onopen')) )
            eval(x + "(dv)");
        classSwap(dv, 'apage','selpage');
        open_divs[realid]=true;
        classSwap(ev, 'atab','seltab');
        vpops[st]=tagid;
    }
}
function popdivQ(ftid)
{
    var e = gE(ftid);
    if( !e ) {
        dbg("missing divid " + ftid);
        return;
    }
    if( e.style.display == 'none' ) {
        e.style.display='block';
    }
    else e.style.display='none';
}
function popdivL(ftid)
{
    var e = gE(ftid);
    if( !e ) {
        dbg("missing object div id " + ftid);
        return;
    }
    if( e.style.display == 'none' ) {
        e.style.display='block';
        if( typeof radLoadDiv == 'function' ) radLoadDiv(e);
    }
    else e.style.display='none';
}
var popdivs=[];
function popdivS(st,ftid,cb)
{
    var e = gE(ftid);
    if( !e ) {
        dbg("missing object id " + ftid);
        return;
    }
    if( typeof cb == 'undefined' ) cb = false;
    if( (st in popdivs) && popdivs[st][0] != false ) {
        var f = gE(popdivs[st][0]);
        f.style.display='none';
        if( popdivs[st][0] == ftid ) {
            if( popdivs[st][1] != false ) popdivs[st][1](popdivs[st][0],false);
            popdivs[st][0] = false;
            return;
        } else {
            if( popdivs[st][1] != false ) popdivs[st][1](popdivs[st][0],ftid);
        }
    }
    popdivs[st] = [ftid,cb];
    e.style.display='block';
}
function popdivI(tgt,newcast,cb)
{
    var e = gE(tgt);
    if( !e ) {
        dbg("missing target id " + tgt);
        return;
    }
    if( typeof cb == 'undefined' ) cb = false;
    if( (tgt in popdivs) ) {
        if( popdivs[tgt][0] == newcast || e.getAttribute('xCast') == newcast ) return;
        if( popdivs[tgt][1] != false ) popdivs[tgt][1](popdivs[tgt][0],newcast);
        clearNode(e);
    }
    popdivs[tgt] = [newcast,cb];
    e.setAttribute('xCast', newcast);
    radLoadDiv(e);
//    blitzTemplate( e, newcast );
}

function safeLink(url,tgt)
{
    var w = window.open();
    w.document.open();
//    w.location.href = url;
    w.document.writeln('<script type="text/javascript">window.location = "http://livehelp.theshopsinsl.com/norefer.php?u=' + url + '";</script>');
    w.document.close();
    w.focus();
}




var pop_menu=false;
function xMenu(obj,ev)
{
    var xid = obj.id + "_buttons";
    var xobj = gE(xid);
    var mp = mc(ev);

    if( pop_menu !== false ) {
        var ol = gE(pop_menu);
        ol.style.display='none';
    }
    xobj.style.display='block';
    xobj.style.position='absolute';
    xobj.style.left = mp[0] + "px";
    xobj.style.top = mp[1] + "px";

    pop_menu=xid;
    setTimeout("xMenuAdjust();",50);
}
function xMenuAdjust()
{
    var xobj = gE(pop_menu);

    if( (parseInt(xobj.style.left)+xobj.clientWidth) > winW ) {
        xobj.style.left = (winW-(xobj.clientWidth+10)) + "px";
    }
    if( (parseInt(xobj.style.top)+xobj.clientHeight) > winH ) {
        xobj.style.top = (winH-(xobj.clientHeight+10)) + "px";
    }

    pop_menu=false;
}

function xDropMenu(obj,ev)
{
    var xid = obj.id + "_buttons";
    var xobj = gE(xid);
    var xp = getPos(obj);

    if( pop_menu !== false ) {
        var ol = gE(pop_menu);
        ol.style.display='none';
    }
    xobj.style.display='block';
    xobj.style.position='absolute';
    xobj.style.left = (xp[0]+15) + "px";
    xobj.style.top = (xp[1]+obj.clientHeight) + "px";
    pop_menu=xid;
}
function xCloseMenu()
{
    if( pop_menu !== false ) {
        var ol = gE(pop_menu);
        ol.style.display='none';
    }

    pop_menu=false;
}








var gform=false, gformi=false;
//registerKeyboard(kbhit);
function useGenericKB()
{
    registerKeyboard(xgenkb);
}
function xgenkb(kk)
{
    if( kk == 13 && gform !== false ) {
        formsend(gformi);
        return 1;
    }
}
function formsend(fx)
{
    var o=fx;
    var x = "";

    for( o = fx; o; o = o.parentNode ) {
      if( o.nodeName == 'FORM' ) {
        break;
      }
    }
    if( !o ) return;

    shellMessage("formsend()");

    if( ( typeof o['cMethod'] != 'undefined' && isValid(x=o['cMethod']) && x != "" ) ||
        ( o.hasAttribute('xMethod') && isValid(x=o.getAttribute('xMethod')  ) && x != "" ) ||
        ( o.hasAttribute('onsubmit') && isValid( x=o.getAttribute('onsubmit') ) && x != "" ) )
    {

    //  if( x.indexOf("return") >= 0 )
    //      x = x.replace(/return/g,'');

        if( x.indexOf("this") >= 0 )
            x = x.replace(/this/g,'o');

        if( x.indexOf("form") >= 0 )
            x = x.replace(/form/g,'o');

        if( x.indexOf("return") >= 0 )
            x = "function __quick_eval() {" + x + "}; __quick_eval();";
        console.info("Running onsend == ", x);
        return eval(x);
    } else if( o.hasAttribute('action') && isValid(x=o.getAttribute('action')) ) {
        console.info("Running onsend()");
        o.submit();
        return false;
    } else return false;

    /*
    if( ( typeof o['cMethod'] == 'undefined' || !isValid(x=o['cMethod']) || x == "" ) &&
        ( !o.hasAttribute('xMethod') || !isValid(x=o.getAttribute('xMethod')  ) || x == "" ) &&
        ( !o.hasAttribute('onsubmit') || !isValid( x=o.getAttribute('onsubmit') ) || x == "" ) )
        return false;

//  if( x.indexOf("return") >= 0 )
//      x = x.replace(/return/g,'');

    if( x.indexOf("this") >= 0 )
        x = x.replace(/this/g,'o');

    if( x.indexOf("frm") >= 0 )
        x = x.replace(/this/g,'o');

    return eval(x);
    */
}
var form_sent=false;
function fsend(eve)
{
    var ev = eve || window.event;
    var o, f, i, found=false;
    form_sent=true;

    if( !ev.parentNode && ev.target ) o = ev.target;
    else o = ev;

    for( ; o; o = o.parentNode ) {
        if( o.nodeName != 'FORM' ) continue;
        for( i in document.forms ) {
            if( document.forms[i] == o ) {
                found=true;
                break;
            }
        }
        if( found ) break;

        if( o.name in document.forms ) {
            found=true;
            break;
        }
    }
    if( !found ) { form_sent=false; return false; }

    shellMessage("fsend()");
    return formsend(o);
}
var gfocusing=false;
var gftx=-1;
var fcel;
function gfocus(ev)
{
    registerKeyboard(xgenkb);
    if( !gfocusing )
        gfocusing=true;
    if( gftx != -1 ) clearTimeout(gftx);
    gftx=setTimeout('gblurok()',100);
    fcel=ev;
    gformi=ev.form;
    var x;
    if( isValid(x = ev.xOldFocus) ) {
        eval(radTranslateFrom(ev,x.replace('this','ev')));
    }
}
function gblur(ev)
{
    if( gfocusing ) return;
    gformi=false;
    unregKeyboard(xgenkb);
    var x;
    if( isValid(x = ev.xOldBlur) ) {
        eval(radTranslateFrom(ev,x.replace('this','ev')));
    }
}
function gblurok()
{
    gftx=-1;
    gfocusing=false;
    //!verify focus (doesn't work)
    //fcel.focus();
}
/*
function fgenform(frm)
{
    var i, x, o, ax=[frm];

    frm.setAttribute('onsubmit','return false;');
    while( ax.length > 0 ) {
        x = ax.splice(0,1)[0];

        for( i=0; i<x.childNodes.length; i++ ) {
            o = x.childNodes[i];
            if( o.nodeName == 'INPUT' ) {
                if( o.type != 'hidden' ) {
                    o.xOldFocus = o.getAttribute('onfocus');
                    o.xOldBlur = o.getAttribute('onblur');
                    o.setAttribute('onfocus', 'gfocus(this)');
                    o.setAttribute('onblur', 'gblur(this)');
                    if( isValid(o.className) )
                        if( o.className.indexOf("textfield") < 0 ) {
                            o.className += " textfield";
                        }
                    else
                        o.className = "textfield";
                }
            } else if( typeof x.childNodes!='undefined' && x.childNodes.length>0 ) {
                ax[ax.length]=o;
            }
        }
    }
}
*/

function fgenform(frm)
{
    var i, x, o, ax=[frm];

//    frm.setAttribute('onsubmit','radSendForm(this); console.log("submit()ed"); return false;');

    while( ax.length > 0 ) {
        x = ax.splice(0,1)[0];

        for( i=0; i<x.childNodes.length; i++ ) {
            o = x.childNodes[i];
            if( o.nodeName == 'INPUT' || o.nodeName == 'SELECT' || o.nodeName == 'TEXTAREA' ) {
                if( o.type != 'hidden' ) {
                    if( o.hasAttribute('onfocus') && o.getAttribute('onfocus') != 'gfocus(this)' ) {
                        o.xOldFocus = o.getAttribute('onfocus');
                    }
                    if( o.hasAttribute('onblur') && o.getAttribute('onblur') != 'gblur(this)' ) {
                        o.xOldBlur = o.getAttribute('onblur');
                    }
                    o.setAttribute('onfocus', 'gfocus(this)');
                    o.setAttribute('onblur', 'gblur(this)');
                    if( isValid(o.className) )
                        if( o.className.indexOf("textfield") < 0 ) {
                            o.className += " textfield";
                        }
                    else
                        o.className = "textfield";
                }
            } else if( typeof x.childNodes!='undefined' && x.childNodes.length>0 ) {
                ax[ax.length]=o;
            }
        }
    }
}






function setHeight(dv,h,pixels)
{
    if( typeof pixels == 'undefined' ) pixels=true;
    dv.style.height = dv.style.maxHeight = h + (pixels ? "px" : "");
}

function scaleDown( img, maxwidth, maxheight )
{
    var w,h;

    w = img.width;
    h = img.height;

    if( w == 0 || h == 0 ) {
        return;
    }

    r = w/h;

    if( w > maxwidth ) {
        w = maxwidth;
        h = w/r;
    }
    if( h > maxheight ) {
        h = maxheight;
        w = h*r;
    }

    img.style.width = w + "px";
    img.style.height = h + "px";
    if( img.style.display == 'none' ) img.style.display = 'block';
}

function indexFrom( msg, ofs, i )
{
    var j,p;

    for( j=0; j<ofs.length; j++ ) {
        if( (p=msg.indexOf(ofs[j],i)) != -1 ) return p;
    }
    return -1;
}

function shortString( str, maxlen )
{
    if( str.length <= maxlen ) return str;
    return str.substr(0,maxlen-3) + "...";
}
function linkify( msg )
{
    var lastpt=0, i=0, len=msg.length, url, startpoint, c;
    var newmsgbuf="", buf;

    msg = "" + msg;

    while( (i=indexFrom(msg, ['http://','https://'],i)) != -1 ) {
        if( lastpt != i )
            newmsgbuf = msg.substr(lastpt,i-lastpt);
        startpoint=i;
        for( i+=6; i<len; i++ ) {
            if( isWhite(msg.substr(i,1)) )
                break;
        }
        if( i >= len ) i = len;
        url = msg.substr(startpoint, (i-startpoint));
        lastpt = i;
        newmsgbuf += "<a href='" + url + "' target=_blank>" + url + "</a>";
    }
    newmsgbuf += msg.substr(lastpt);
    return newmsgbuf;
}

var plots={};
// function scalarAnima(), generates coordinated callbacks
// eg scalarTimer( animator, 5000, 100, { 'height1': [ 0, 200 ], 'height2': [ 100, 110 ] } );
// will call animator({'height1':x,'height2':y});
function scalarAnima( fn, totalTime, timeStep, targets )
{
    var r=randStr(5);
    var x={'f':fn,'r':r,'ts':timeStep,'tt':Math.ceil(totalTime/timeStep),'tn':0};
    for( var i in targets ) {
        targets[i][2] = ( targets[i][1] - targets[i][0] )  / x.tt; // iteration value
    }
    x.tx = targets;
    plots[r]=x;
    plots[r].i = setInterval('_scalarTimer("'+x.r+'")',timeStep);
}
function _scalarTimer( r )
{
    var x = plots[r];
    x.tn++;
    if( x.tn >= x.tt ) {
        clearInterval( x.i );
        for( var i in x.tx ) {
            plots[r].tx[i][0] = x.tx[i][1];
        }
    } else {
        for( var i in x.tx ) {
            plots[r].tx[i][0] += x.tx[i][2];
        }
    }
    x.f(plots[r].tx);
    if( x.tn >= x.tt ) {
        delete plots[r];
    }
}

/* todo: rewrite to use maxRange calculation to calculate totalTime from maxMove
var plots={};
// function scalarTimer(), generates coordinated callbacks
// eg scalarTimer( animator, 5000, 100, { 'height1': [ 0, 200 ], 'height2': [ 100, 110 ] } );
// will call animator({'height1':x,'height2':y});
function scalarTimer( fn, timeStep, maxMove, targets )
{
    var r=randStr(5);
    var x={'r':r,'ts':timeStep,'tt':totalTime};
    var i, maxRange, maxRangeN=-1;
    for( i in targets ) {
        targets[i][2] = targets[i][0]; // current value
        targets[i][3] = Math.abs(targets[i][1] - targets[i][0]); // total range
        if( targets[i][3] > maxRange ) {
            maxRange = targets[i][3];
            maxRangeN = i;
        }
    }
    var maxIter = maxRange/timeStep;
    for( i in targets ) {
        targets[i][4] = ( targets[i][3] / timeStep ) * maxIter;
    }

    x.tx = targets;
    x.i = setInterval('_scalarTimer("'+x.r+'")',timeStep);
}
function _scalarTimer( r )
{
    var x = plots[r];
}
*/

var as1 = []; // for parameters eg scrollTop
function aniScroll1( id, parm, tgt, rate, speed )
{
    var e = gE(id);
    var cur = parseFloat(e[parm]);
    var vid = id;
    while( vid in as1 ) {
        vid += "_" + randStr(3);
    }
    as1[vid] = [ id, parm, cur, rate, tgt, setInterval('_aniScroll1("' + vid + '")', speed) ];
}
function _aniScroll1( vid )
{
    var x = as1[vid];
    var e = gE( x[0] );
    var done=false;
    x[2] += x[3];
    if( ( x[3] > 0 && x[2] >= x[4] ) ||
        ( x[3] < 0 && x[2] <= x[4] ) ) {
        clearInterval( x[5] );
        x[2] = x[4];
        done=true;
    }
    e[x[1]] = x[2];
    if( done )
        delete as1[vid];
}


var as2 = []; // for styles
function aniScroll2( id, parm, tgt, rate, speed )
{
    var e = gE(id);
    var cur = parseFloat(getStyle(e,parm));
    var vid = id;
    while( vid in as2 ) {
        vid += "_" + randStr(3);
    }
    as2[vid] = [ id, parm, cur, rate, tgt, setInterval('_aniScroll2("' + vid + '")', speed) ];
}
function _aniScroll2( vid )
{
    var x = as2[vid];
    var e = gE( x[0] );
    var done=false;
    x[2] += x[3];
    if( ( x[3] > 0 && x[2] >= x[4] ) ||
        ( x[3] < 0 && x[2] <= x[4] ) ) {
        clearInterval( x[5] );
        x[2] = x[4];
        done=true;
    }
    e[x[1]] = x[2] + "px";
    if( done )
        delete as2[vid];
}
