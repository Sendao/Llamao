
function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1) ) + min;
}

class Shimm
{

	constructor(shi, m)
	{
		this.m = this.lowercaseKeys(m);
		this.l = {};
		this.shi = shi;
	}

	lowercaseKeys( m )
	{
		var o = {};

		for( var k in m ) {
			if( typeof m[k] == 'object' ) {
				if( Array.isArray(m[k]) )
					o[k.toLowerCase()] = this.lowercaseKeysArr(m[k]);
				else
					o[k.toLowerCase()] = this.lowercaseKeys(m[k]);
			} else {
				o[k.toLowerCase()] = m[k];
			}
		}

		return o;
	}
	lowercaseKeysArr( m )
	{
		var o = [];

		for( var i=0; i<m.length; i++ ) {
			if( typeof m[i] == 'object' ) {
				if( Array.isArray(m[i]) )
					o.push(this.lowercaseKeysArr(m[i]));
				else
					o.push(this.lowercaseKeys(m[i]));
			} else {
				o.push(m[i]);
			}
		}

		return o;
	}

	request( location )
	{
		if( location == "" || typeof location != 'string' ) return [null,this.m];
		let d = location.split('.');
		var x = this.m, p = null;
		for( var i=0; i<d.length; i++ ) {
			if( !(d[i] in x) ) return [null,this.m];
			p = x;
			x = x[d[i]];
		}
		return [p,x];
	}

	look( nest, data )
	{
		return nest[data.toLowerCase()];
	}
	check( nest, data ) 
	{
		return( (data.toLowerCase() in nest) );
	}

	travel( ch, dest )
	{
		let place = ( ch in this.l ) ? this.l[ch] : '';
		let cmds, up;
		[up,cmds] = this.request(place);
		if( cmds === this.m ) place = "";

		if( this.check(cmds, dest) ) {
			cmds = this.look(cmds, dest);
			if( this.m !== cmds ) place += "." + dest;
			else place = "";
		} else if( this.check(this.m, dest) ) {
			cmds = this.look(this.m, dest);
			place = dest;
		} else {
			cmds = this.m;
			place = "";
		}
		//console.log(ch + " to " + dest + ": " + typeof cmds, cmds);

		if( typeof cmds == 'object' ) {
			this.l[ch] = place;
			if( 'text' in cmds ) {
				let msg = "";

				if( typeof cmds.text == 'string' ) {
					msg = cmds.text;
				} else {
					let n = randomInt(0, cmds.text.length-1);
					msg = cmds.text[n];
				}
				this.shi.sendActor( ch, 'MVerse', (place != "" ? ( place + "=>" ) : "((Welcome))" ) + msg.replaceAll("%1", ch) );
			}

			if( 'next' in cmds ) {
				this.shi.moveActor( ch, cmds.next );
			}
		} else {
			this.shi.sendActor( ch, 'MVerse', (place != "" ? ( place + "=>" ) : "((Welcome))" ) + cmds.replaceAll("%1", ch) );
		}
	}

	run( ch, tokens )
	{
		let place = ( ch in this.l ) ? this.l[ch] : '';
		let cmds, up;
		[up,cmds] = this.request(place);
		if( cmds === this.m ) place = "";

		if( typeof tokens == 'string' ) tokens = tokens.toLowerCase().split(" ");

		for( var i=0; i<tokens.length; i++ ) {
			if( this.check(cmds, tokens[i]) ) {
				this.shi.moveActor( ch, tokens[i] );
				break;
			}
		}
	}

}



module.exports = { Shimm };