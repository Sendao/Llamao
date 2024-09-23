function selectTab(div)
{
    var tabholder, resourceid;
    var n;
    
    if( (tabholder=div.parentNode) ) {
        if( (n=tabholder.id.indexOf('_tabholder')) != -1 ) {
            resourceid = tabholder.id.substr(0, n);
            
            if( tabholder.focusedTab ) {
                classSwap( tabholder.focusedTab, 'hightab', 'lowtab' );
            }
            tabholder.focusedTab = div;
        } else {
            console.log("couldn't find tabholder");
        }
    }
    if( div.className.includes('lowtab') ) {
        classSwap(div, 'lowtab', 'hightab');
    }
    if( !tabholder.bodyArea ) {
        var n=0;
        //while( n < tabholder.)
    }
}

var clx = {
    'lowtab': { 'onclick': 'selectTab(this)' },
    'hightab': { 'onclick': '' }
};

//use: radClass(clx);
//Want to change an object's class? No problem: use classSwap() from lib.js
//classSwap( div, oldclass, newclass );
//and afterward scan the class to update its functionality:
//radScanClass(div);


function openWindow(tpl, root)
{
    if( typeof root == 'undefined' ) root = '';
    var e = genTemplate(tpl,root);
    e.className = 'br3 g0';
    e.id = 'window_' + randStr(6);
    e.style.position = 'absolute';
    e.style.left = '0px';
    e.style.top = winH + 'px';
    e.style.zIndex = '-1';
    e.style.maxWidth = (winW-40)+"px";
    e.style.minWidth = "200px";
    e.style.minHeight = "48px";
    e.style.border = '1px solid tan';
    e.style.opacity = '1.0';
    e.style.background = 'black';
//    e.style. curve
    
    document.body.appendChild(e);
    setTimeout( "viewWindow('"+e.id+"');", 20);
}
function openWindow2(tpl, root, obj)
{
    radStore(root,obj);
    openWindow(tpl,root);
}
function viewWindow(eid)
{
    var e = gE(eid);
    e.setAttribute('xPostDraw','omv(["page","abs","bot","center","offy",-124],0,-' + e.clientWidth + ',0,-' + (e.clientHeight+4) + ')');
    radProcess(e, document.body);
    e.style.zIndex = '100';
}
function closeWindow(dv)
{
    var e = dv;

    while( e && e.id.indexOf("window_") < 0 ) {
        e = e.parentNode;
    }
    if( !e ) {
        dbg("Can't find element to close");
        return;
    }

    kamiNode(e);
}

var full_log = [];
function finishLoad()
{   
    var bl = radVar("botlog");
    var i;
    if( bl == null ) {
        radStore("botlog",[]);
        bl=[];
    }
    for( i=0; i<bl.length; i++ ) {
        full_log.push( [2, bl[i]] );
    }
    radStore("full_log",full_log);
//    streamNext();
//    document.onkeydown = dcKB;
    document.body.style.overflow = 'hidden';
	radStore("cmdlog", []);
	clx = {
		'lowtab': { 'onclick': 'selectTab(this)' },
		'hightab': { 'onclick': '' },
		'wgtitlebar': { 'onmousedown': 'drag_glass(event)' },
		'wgresize': { 'onmousedown': 'size_glass(event)' }
	};
	radClass(clx);
	radFactorable("projects",1);
}

function scriptResponse( obj )
{
    // obj = { code, msg, text(msg), qry(r), act(page), cmd(m) }
    full_log.unshift( [1, obj] );
    radStore("full_log", full_log);
}


function updList(ln,ix,idf)
{
	var i,j;
	var xi = radVar(ln);
	var addc=0,updc=0;

	if( xi === null ) {
		if( idf === true || idf === false ) xi = [];
		else xi = {};
	}
	if( typeof idf == 'undefined' ) idf = 'id';
	if( idf === false ) {
		for( i=ix.length; i>=0; i-- ) {
			if( (j=xi.indexOf(ix[i])) >= 0 )
				xi.splice(j,1);
			xi.unshift(ix[i]);
		}
	} else {
		for( i in ix ) {
			if( idf === true ) {
				if( (j=xi.indexOf(ix[i])) >= 0 )
					xi.splice(j,1);
				xi.push(ix[i]);
				updc++;
			} else {
				xi[ ix[i][idf] ] = ix[i];
				addc++;
			}
		}
	}
	shellMessage(ln + ":" + updc + "+" + addc);

	radStore(ln, xi);
	radLoadSect(ln);
}

function solveReference( mainlist, refid, itemid )
{
	var xl = radVar(mainlist);
	var i;
	if( radVar("cdesk") == refid ) radStore("cdesk", itemid);

	for( i in xl ) {
		if( xl[i].refid == refid || xl[i].id == refid || xl[i].gid == refid ) {
			radRefactor( { 'root': mainlist, 'ident': mainlist + "." + i }, { 'root': mainlist + "." + itemid } );
			delete xl[i];
		}
	}

	if( mainlist == 'items' )
		deskReplaceItem(refid,itemid);
}




var omdivs=[];
//omv
//loc=[page|box,top|mid|bot,left|center|right,abs|rel]
//xs=1: full width
//xs=1,xr=20: full width - 20
//xs=0,xr=20: full width - 20
//xs=0,xr=-20: 20
//xs=0.5: half width
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

    mm.setAttribute('resizing','1');

	XRegisterResize('autoOMV2("' + mm.id + '")');
//	if( mm.style.display != 'none' ) {
		setTimeout('autoOMV2("' + mm.id + '")',1);
//	} else {
//		waitForVis(mm.id,'autoMSize2("' + mm.id + '")');
//	}
}
function autoOMV2(divid)
{
	if( !isValid(divid) ) { alert("missing div id"); return -1; }
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
    var offx=0,offy=0;
	var userel=false,useabs=false,usenul=true;
    var saveAs=false,nosize=false;

	var pxn=mm.parentNode;
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
			case 'middle': case 'mid': ya=0; break;
			case 'rel': case 'relative': userel=true; break;
			case '!rel': case '!relative': userel=false; break;
			case 'abs': case 'absolute': useabs=true; break;
			case '!abs': case '!absolute': useabs=false; break;
			case 'nul': case 'inline': usenul=true; break;
            case 'offx': i++; offx = locs[i]; break;
            case 'offy': i++; offy = locs[i]; break;
            case 'save': i++; saveAs = locs[i]; break;
            case 'ns': nosize=true; break;
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
//	if( xp != 0 ) ww -= xp;
//	if( xm != 0 ) ww -= xm;

	if( pxn != document.documentElement ) {
        if( pxn.getAttribute('resizing') == '1' ) {
            setTimeout('autoOMV2("'+divid+'")',100);
            return -1;
        }
		ww = pxn.clientWidth;
		hh = pxn.clientHeight;
        var pxnB=pxn.parentNode;
        while( pxnB && ( hh <= 0 || ww <= 0 ) ) {
            if( hh <= 0 ) hh = pxnB.clientHeight;
            if( ww <= 0 ) ww = pxnB.clientWidth;
            pxnB = pxnB.parentNode;
        }
	}
    mm.setAttribute('resizing','0');

	if( !isNaN(xs) && xs != 0 ) tw = ww*xs;
	if( !isNaN(xr) && xr != 0 ) {
		if( xs != 0 ) tw -= xr;
		else if( xr > 0 ) tw = ww-xr;
		else tw = -xr;
	}

	if( tw !== false && tw >= 0 ) {
		tw -= (xm+xp);
        if( !nosize )
    		mm.style.maxWidth = mm.style.width = tw + "px";
		//mm.clientWidth = tw;
	}

//	if( yp != 0 ) hh -= yp;
//	if( ym != 0 ) hh -= ym;
	if( ys != 0 ) th = hh*ys;
	if( yr != 0 ) {
		if( ys != 0 ) th -= yr;
		else if( yr > 0 ) th = hh-yr;
		else th = -yr;
	}

	if( th !== false && th >= 0 ) {
		th -= (ym+yp);
        if( !nosize )
    		mm.style.maxHeight = mm.style.height = th + "px";
		//mm.clientHeight = th;
	}

    var saveData = { 'w': tw, 'h': th };
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
		if( !isNaN(n) ) {
            saveData.x = n+offx;
			mm.style.left = (n+parseInt(offx)) + "px";
        }

		if( ya === false ) {
			n = xy[1];
		} else if( ya < 0 ) {
			n = 0;
		} else if( ya == 0 ) {
			n = (hh-(th+ym+yp))/2;
		} else {
			n = hh-(th+ym+yp);
		}
		if( !isNaN(n) ) {
            saveData.y = n+offy;
			mm.style.top = (n+parseInt(offy)) + "px";
        }
	}

    if( saveAs !== false ) {
        radCStore(saveAs, saveData);
    }
	return 0;
}