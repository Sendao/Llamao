
class MinHeap {
    constructor(max, valname='val', idxname='pos')
    {
        if( typeof max == 'undefined' ) {
            this.maxsize = 32;
        } else {
            this.maxsize = max;
        }
        this.size = 0;
        this.heap = new Array(this.maxsize);
        this.vx = valname;
        this.ix = idxname;
    }

    print() {
        let i;
        let buf = "";
        var l,r;

        console.log("Printing MinHeap: Maxsize " + this.maxsize + ", Size " + this.size);

        for( i=0; i<this.size/2; i++ ) {
            buf = "Parent: " + this.heap[i][this.vx];
            l = this.left(i); r = this.right(i);
            if( l < this.size && this.heap[l][this.vx] != Infinity )
                buf += " Left: " + this.heap[l][this.vx];
            if( r < this.size && this.heap[r][this.vx] != Infinity )
                buf += " Right: " + this.heap[r][this.vx];
            console.log(buf);
        }
    }

    go(n) {
        let l,r;
        if( this.leaf(n) ) return;
        
        l = this.left(n);
        r = this.right(n);
        //console.log("go(" + n + ") (" + l + "," + r + ") " + this.size);
        if( l >= this.size ) {
            if( r >= this.size ) {
                return;
            }
            if( this.heap[n][this.vx] > this.heap[r][this.vx] ) {
                this.swap(n, r);
                this.go(r);
            }
        } else if( r >= this.size ) {
            if( this.heap[n][this.vx] > this.heap[l][this.vx] ) {
                this.swap(n, l);
                this.go(r);
            }
        } else if( this.heap[n][this.vx] > this.heap[l][this.vx] ||
            this.heap[n][this.vx] > this.heap[r][this.vx] ) {
            if( this.heap[l][this.vx] < this.heap[r][this.vx] ) {
                this.swap(n, l);
                this.go(l);
            } else {
                this.swap(n, r);
                this.go(r);
            }
        }
    }

    parent(index) { return parseInt((index-1)/2); }
    left(index) { return parseInt(2*index)+1; }
    right(index) { return parseInt(2*index)+2; }
    leaf(index) { return (2*index)+1>=this.size && index < this.size; }

    swap(a,b) {
        [this.heap[a], this.heap[b]] = [this.heap[b], this.heap[a]];        
        this.heap[a][this.ix] = a;
        this.heap[b][this.ix] = b;
    }

    removeIdx(n) {
        var l,r;
        if( n >= this.size ) return;
        if( this.leaf(n) ) {
            this.heap[n] = {}; this.heap[n][this.vx] = Infinity; this.heap[n][this.ix] = n;
            while( this.size > 0 && this.heap[ this.size-1 ][this.vx] == Infinity ) this.size--;
            return;
        }

        l = this.left(n);
        r = this.right(n);

        if( l >= this.size ) {
            this.heap[n] = this.heap[r];
            this.heap[n][this.ix] = n;
            this.removeIdx(r);
        } else if( r >= this.size ) {
            this.heap[n] = this.heap[l];
            this.heap[n][this.ix] = n;
            this.removeIdx(l);
        } else if( this.heap[l][this.vx] < this.heap[r][this.vx] ) {
            this.heap[n] = this.heap[l];
            this.heap[n][this.ix] = n;
            this.removeIdx(l);
        } else {
            this.heap[n] = this.heap[r];
            this.heap[n][this.ix] = n;
            this.removeIdx(r);
        }
    }

    remove(num) {
        let stack = [0];
        var i;

        while( stack.length > 0 ) {
            i = stack.shift();

            if( this.heap[i][this.vx] == num ) {
                this.removeIdx(i);
                return true;
            }
            stack.push( this.left(i) );
            stack.push( this.right(i) );
        }
        return false;
    }

    removeAll(num) {
        while( this.remove(num) ) continue;
    }

    push(obj)
    {
        if( this.size+1 >= this.maxsize ) {
            this.heap = this.heap.concat( new Array(this.maxsize) );
            this.maxsize *= 2;
        }
        obj[this.ix] = this.size;
        this.heap[this.size] = obj;

        let i = this.size;
        let p = this.parent(i);
        while( this.heap[i][this.vx] < this.heap[p][this.vx] ) {
            this.swap(i, p);
            i = p;
            p = this.parent(i);
        }
        this.size++;
    }

    pop()
    {
        if( this.size == 0 ) return Infinity;
        let v = this.heap[0];
        this.size--;
        if( this.size == 0 ) {
            this.heap[0] = {}; this.heap[0][this.vx] = Infinity; this.heap[0][this.ix] = 0;
            return v;
        }
        this.heap[0] = this.heap[this.size];
        this.heap[0][this.ix] = 0;
        this.heap[this.size] = {}; this.heap[this.size][this.vx] = Infinity; this.heap[this.size][this.ix] = this.size;
        if( this.size != 1 )
            this.go(0);
        return v;
    }

    peek()
    {
        if( this.size == 0 ) return Infinity;
        let v = this.heap[0];
        return v;
    }
}

let UFind = function(n) {    
    this.parent = [];
    this.rank = [];
    
    for( var i=0; i<n; i++ ) {
        this.parent.push(i);
        this.rank.push(0);
    }

    this.find = function(x) {
        if( this.parent[x] != x ) {
            this.parent[x] = this.find(this.parent[x]);
        }
        return this.parent[x];
    }
    
    this.union = function(x,y) {
        let px = this.find(x);
        let py = this.find(y);

        if( px != py ) {
            if( this.rank[px] < this.rank[py] ) {
                this.parent[px] = py;
            } else if( this.rank[px] > this.rank[py] ) {
                this.parent[py] = px;
            } else {
                this.parent[py] = px;
                this.rank[px] ++;
            }
        }
    }

    this.connected = function(x,y) {
        return ( this.find(x) == this.find(y) );
    }
};

class MaxHeap {

    constructor(max, valname='val', idxname='pos')
    {
        if( typeof max == 'undefined' ) {
            this.maxsize = 32;
        } else {
            this.maxsize = max;
        }
        this.size = 0;
        this.heap = new Array(this.maxsize);
        this.vx = valname;
        this.ix = idxname;
    }

    print() {
        let i;
        let buf = "";
        var l,r;

        console.log("Printing MaxHeap: Maxsize " + this.maxsize + ", Size " + this.size);

        for( i=0; i<this.size/2; i++ ) {
            buf = "Parent: " + this.heap[i][this.vx];
            l = this.left(i); r = this.right(i);
            if( l < this.size && this.heap[l][this.vx] != -Infinity )
                buf += " Left: " + this.heap[l][this.vx];
            if( r < this.size && this.heap[r][this.vx] != -Infinity )
                buf += " Right: " + this.heap[r][this.vx];
            console.log(buf);
        }
    }

    go(n) {
        let l,r;
        if( this.leaf(n) ) return;
        
        l = this.left(n);
        r = this.right(n);
        if( l >= this.size ) {
            if( r >= this.size ) {
                return;
            }
            if( this.heap[n][this.vx] < this.heap[r][this.vx] ) {
                this.swap(n, r);
                this.go(r);
            }
        } else if( r >= this.size ) {
            if( this.heap[n][this.vx] < this.heap[l][this.vx] ) {
                this.swap(n, l);
                this.go(l);
            }
        } else if( this.heap[n][this.vx] < this.heap[l][this.vx] ||
            this.heap[n][this.vx] < this.heap[r][this.vx] ) {
            if( this.heap[l][this.vx] > this.heap[r][this.vx] ) {
                this.swap(n, l);
                this.go(l);
            } else {
                this.swap(n, r);
                this.go(r);
            }  
        }
    }

    parent(index) { return parseInt((index-1)/2); }
    left(index) { return parseInt(2*index)+1; }
    right(index) { return parseInt(2*index)+2; }
    leaf(index) { return (2*index)+1>=this.size && index < this.size; }

    swap(a,b) {
        [this.heap[a], this.heap[b]] = [this.heap[b], this.heap[a]];        
        this.heap[a][this.ix] = a;
        this.heap[b][this.ix] = b;
    }

    removeIdx(n) {
        var l,r;
        if( n >= this.size ) return;
        if( this.leaf(n) ) {
            this.heap[n] = {}; this.heap[n][this.vx] = -Infinity; this.heap[n][this.ix] = n;
            while( this.size > 0 && this.heap[ this.size-1 ][this.vx] == -Infinity ) this.size--;
            return;
        }

        l = this.left(n);
        r = this.right(n);

        if( l >= this.size ) {
            this.heap[n] = this.heap[r];
            this.heap[n][this.ix] = n;
            this.removeIdx(r);
        } else if( r >= this.size ) {
            this.heap[n] = this.heap[l];
            this.heap[n][this.ix] = n;
            this.removeIdx(l);
        } else if( this.heap[l][this.vx] > this.heap[r][this.vx] ) {
            this.heap[n] = this.heap[l];
            this.heap[n][this.ix] = n;
            this.removeIdx(l);
        } else {
            this.heap[n] = this.heap[r];
            this.heap[n][this.ix] = n;
            this.removeIdx(r);
        }
    }

    remove(num) {
        let stack = [0];
        var i;

        while( stack.length > 0 ) {
            i = stack.shift();

            if( this.heap[i][this.vx] == num ) {
                this.removeIdx(i);
                return;
            }
            stack.push( this.left(i) );
            stack.push( this.right(i) );
        }
        //! not found
    }

    push(obj)
    {
        if( this.size+1 >= this.maxsize ) {
            this.heap = this.heap.concat( new Array(this.maxsize) );
            this.maxsize *= 2;
        }
        obj[this.ix] = this.size;
        this.heap[this.size] = obj;

        let i = this.size;
        let p = this.parent(i);
        while( this.heap[i][this.vx] > this.heap[p][this.vx] ) {
            this.swap(i, p);
            i = p;
            p = this.parent(i);
        }
        this.size++;
    }

    pop()
    {
        if( this.size == 0 ) return 0;
        let v = this.heap[0];
        this.size--;
        if( this.size == 0 ) {
            this.heap[0] = {}; this.heap[0][this.vx] = -Infinity; this.heap[0][this.ix] = 0;
            return v;
        }
        this.heap[0] = this.heap[this.size];
        this.heap[0][this.ix] = 0;
        this.heap[this.size] = {}; this.heap[this.size][this.vx] = -Infinity; this.heap[this.size][this.ix] = this.size;
        if( this.size != 1 )
            this.go(0);
        return v;
    }

    peek()
    {
        if( this.size == 0 ) return -Infinity;
        let v = this.heap[0];
        return v;
    }
}

class MemPool {
    constructor( ) {
        this.freed = [];
        this.maxfreed = 0;
    };

    release(ptr) {
        this.freed.push(ptr);
        if( this.freed.length > this.maxfreed ) this.maxfreed = this.freed.length;
    }

    releaseAll(arr) {
        this.freed = this.freed.concat(arr);
        if( this.freed.length > this.maxfreed ) this.maxfreed = this.freed.length;
    }

    get(params) {
        var v, i;
        if( this.freed.length > 0 ) {
            v = this.freed.shift();
            if( typeof params != 'undefined' ) {
                for( var i in params ) {
                    v[i] = params[i];
                }
            }
            return v;
        }

        v = {};
        if( typeof params != 'undefined' ) {
            for( i in params ) {
                v[i] = params[i];
            }
        }
        return v;
    };
}

class ArrPool {
    constructor( ) {
        this.freed = [];
        this.maxfreed = 0;
    };

    release(ptr) {
        this.freed.push(ptr);
        if( this.freed.length > this.maxfreed ) this.maxfreed = this.freed.length;
    }

    releaseAll(arr) {
        this.freed = this.freed.concat(arr);
        if( this.freed.length > this.maxfreed ) this.maxfreed = this.freed.length;
    }

    get(params) {
        var v, i;
        if( this.freed.length > 0 ) {
            v = this.freed.shift();
            if( typeof params != 'undefined' ) {
                for( i=0; i<params.length; i++ ) {
                    v[i] = params[i];
                }
            }
            return v;
        }

        v = [];
        if( typeof params != 'undefined' ) {
            for( i=0; i<params.length; i++ ) {
                v[i] = params[i];
            }
        }
        return v;
    };
}

module.exports = { MinHeap, MaxHeap, MemPool };
