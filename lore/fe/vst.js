
function requestForm( fm, jobcmd, cb )
{
	var jc = randStr(4);
	var i,len=fm.length,found=false;
	for( i=0; i<len; i++ ) {
		if( fm[i].name == "jc" ) {
			if( fm[i].value != "" ) {
				finishStatus(fm[i].value);
			}
			fm[i].value = jc;
			found=true;
			break;
		}
	}
	if( !found )
		aC(fm,cHidden('jc',jc));

	startStatus(jc,"Requesting " + jobcmd);
	VstForm(fm, cb);
}

function runForm( fm, jobcmd, cb )
{
	var jc = randStr(4);
	var i,len=fm.length,found=false;
	for( i=0; i<len; i++ ) {
		if( fm[i].name == "jc" ) {
			if( fm[i].value != "" ) {
				finishStatus(fm[i].value);
			}
			fm[i].value = jc;
			found=true;
			break;
		}
	}
	if( !found )
		aC(fm,cHidden('jc',jc));

	startStatus(jc,jobcmd);
	VstForm(fm, cb);
}

function runJob( phf, php, jobcmd, cbupd, cbfin )
{
	var jc = randStr(4);
	if( jobcmd != "" ) {
		if( php != "" ) php += "&";
		php += "jc="+jc;
		startStatus(jc,"Requesting " + jobcmd);
	}
	if( sch_started == 0 )
		openJobScheduler();
	VstRequest(phf, php);
}
function runTask( phf, php, jobcmd, cbupd, cbfin )
{
	var jc = randStr(4);
	if( jobcmd != "" ) {
		if( php != "" ) php += "&";
		php += "jc="+jc;
		startStatus(jc,jobcmd);
	}
	VstRequest(phf, php);
}
function runPull( phf, jobcmd )
{
	var jc = randStr(4);
	if( jobcmd != "" ) {
		phf += "&jc="+jc;
		startStatus(jc,jobcmd);
	}
	getVst(phf);
}



var statusL = [];
var mainstatus = "";

function errorLog(code, msg)
{
	updateStatus(code,2,msg,"error");
}
function appLog(code, msg)
{
	updateStatus(code,1,msg);
}
function finishStatus(code)
{
	updateStatus(code,5,"Success","done");
}

function msgLog(msg)
{
	addStatus(msg);
}

function startStatus(code, msg)
{
	statusL.push( [code,0,msg] );
	mainstatus = msg + " ...";
	verifyStatus();

}
var status_ticker=-1;
function addStatus(msg)
{
	statusL.push( ["nocode",0,msg] );
	mainstatus=msg;
	verifyStatus();
}
function updateStatus(code,status,msg,aug)
{
	var i,len=statusL.length;
	var found=false;
	for(i=len-1;i>0;i--){
		if( statusL[i][0] == code ) {
			if( i != len-1 ) {
				var xwerthy = statusL[len-1];
				statusL[len-1]=statusL[i];
				statusL[i]=xwerthy;
				statusL[len-1][1] = status;
				statusL[len-1].push(msg);
				found=true;
				break;
			}
		}
	}
	if( !found ) {
		statusL.push( [code,status,"Unknown",msg] );
		len++;
	}
	mainstatus = statusL[len-1][2] + ( aug != "" ? " " + aug : "" ) + ": " + msg;
	verifyStatus();
}


var statbarmax=0;
function verifyStatus()
{
	if( statbarmax == 0 ) {
		statbarmax=statusL.length;
		if( statbarmax > 5 ) statbarmax = 5;
	}
	radStore("currentstatus", mainstatus);
	radStore("statuslines", statusL.reverse());
/*	var statbar = gE("statusbar");
	if( !statbar ) {
		statbar = copyTemplate("statusbar");
		statbar.id="statusbar";
		radLoadSect("status");
		tailHook("Status",statbar);
	} else {
		radLoadSect("status");
		setTimeout("tailFix()",1);
	}*/
	updStatbarMax(3000);
}

function updStatbarMax(set_timer)
{
	setTimeout("decrStatbarMax();", set_timer);
	var g = gE("statusbarbody");
	if( g ) {
		g.setAttribute("xMax", statbarmax);
		g.xMax=statbarmax;
		radLoadDiv(g);
	}
}
function decrStatbarMax()
{
	if( statbarmax > 0 ) statbarmax--;
	if( statbarmax > 3 ) {
		updStatbarMax(500);
		return;
	} else if( statbarmax > 0 ) {
		updStatbarMax(1000);
	} else {
		clearStatus();
	}
}
function clearStatus()
{
	mainstatus = "";
	radClear("currentstatus");
	//var statbar=gE("statusbar");
	radClear("statuslines");
	//kamiNode(statbar);
	//setTimeout("tailFix()",1);
}
