typedef struct Sparse_list sparse_list;
typedef struct Sparse_patch sparse_patch;
struct Sparse_list {
    int offset;
    int size;
    void *data;
    sparse_list *next;

    Sparse_list(int target=0, int sz=0, void *ptr=NULL)
    {
        offset = target;
        size = sz;
        data = ptr;
        next = NULL;
    }

    sparse_list *copy( void )
    {
        sparse_list *head=NULL, *i, *b, *last=NULL;
        void *cpy;

        for( i = this; i; i = i->next ) {
            cpy = malloc( i->size );
            memcpy( cpy, i->data, i->size );
            b = new sparse_list(i->offset, i->size, cpy );
            if( last ) last->next = b;
            else if( !head ) head=b;
            last=b;
        }
        return head;
    }

    sparse_list *merge( sparse_list *b, bool inplace=false )
    {
        sparse_list *tgt = inplace ? this : this->copy();

        for( sparse_list *i = b; i; i = i->next ) {
            tgt->append( i->offset, i->size, i->data );
        }

        return tgt;
    }

    void apply( void *target )
    {
        uint8_t *tgtbuf=(uint8_t*)target;
        for( sparse_list *x = this; x; x = x->next ) {
            if( x->size > 0 )
                memcpy( tgtbuf+x->offset, x->data, x->size );
        }
    }

    uint8_t *save( int *out_sz )
    {
        uint8_t *buf;
        sparse_list *x = this;
        int sz=0;

        for( x = this; x; x = x->next ) {
            sz += sizeof(int)*2 + sizeof(float)*x->size;
        }
        *out_sz = sz;
        buf = (uint8_t*)malloc(sz);
        uint8_t *ptr=buf;
        for( x = this; x; x = x->next ) {
            memcpy( ptr, &x->offset, sizeof(int) );
            memcpy( ptr+sizeof(int), &x->size, sizeof(int) );
            memcpy( ptr+2*sizeof(int), x->data, x->size );
            ptr += 2*sizeof(int) + x->size;
        }

        return buf;
    }
    void load(uint8_t *buf, int bufsz)
    {
        sparse_list *x;
        int off, offset, size;
        void *data;

        for( off=0; off<bufsz; off += 2*sizeof(int) + size ) {
            memcpy(&offset, buf+off, sizeof(int));
            memcpy(&size, buf+off+sizeof(int), sizeof(int));
            append( offset, size, buf+off+sizeof(int)*2 );
        }
    }

    void append( int target, int sz, void *data )
    {
        sparse_list *x=this, *y, *last;

        int ext_sz=(sz+2)*3;

        do {
            if( target <= x->offset+x->size*2 ) {
                x->joindata( target, sz, data );
                if( x->next ) {
                    int endpt = x->offset + x->size;
                    while( x->next ) {
                        if( endpt < x->next->offset ) break;
                        if( x->next->offset + x->next->size > endpt ) {
                            int skipamt = endpt - x->next->offset;
                            int newsz = x->next->size - skipamt;
                            void *newptr = malloc( newsz );
                            memcpy( newptr, (void*)((char*)x->next->data + skipamt), newsz );
                            free(x->next->data);
                            x->next->data = newptr;
                            break;
                        } else {
                            free(x->next->data);
                            x->next = x->next->next;
                        }
                    }
                }
                return;
            }
            if( !x->next ) {
                y = new sparse_list(target,sz,malloc(sz));
                memcpy( y->data, data, sz );
                x->next = y;
                return;
            } else if( target < x->next->offset ) {
                if( target+ext_sz > x->next->offset ) {
                    int prebufsz = (x->next->offset-target);
                    int newsz = x->next->size + prebufsz;
                    void *newptr = NULL;
                    if( sz > newsz ) {
                        newsz = sz;
                        newptr = malloc( sz );
                        memcpy(newptr, data, sz);
                    } else {
                        newptr = malloc( newsz );
                        memcpy((void*)((char*)newptr+prebufsz), x->next->data, newsz - prebufsz );
                        memcpy(newptr, data, sz);
                        if( sz + x->next->size < newsz ) {
                            memset((void*)((char*)newptr+sz), 0, newsz - ( sz + x->next->size ) );
                        }
                    }

                    x->next->size = newsz;
                    x->next->offset = target;

                    free(x->next->data);
                    x->next->data = newptr;
                    return;
                }
                y = new sparse_list(target,sz,malloc(sz));
                memcpy( y->data, data, sz );
                y->next = x->next;
                x->next = y;
                return;
            }
        } while( (x = x->next)!=NULL );
    }

    void joindata( int target, int sz, void *newdata )
    {
        int total_size = (target-offset)+sz;
        void *newptr = malloc( total_size );
        int target_offset = target-offset;
        int old_size = total_size-sz;
        memcpy(newptr, data, old_size > size ? size : old_size );
        if( old_size > size )
            memset((void*)((char*)newptr+size), 0, old_size-size);
        memcpy((void*)((char*)newptr+target_offset), newdata, sz);
        free(data);
        data = newptr;
        size = total_size;
    }

    void report(bool kv, int il)
    {
        uint32_t sum=0, cnt=0, max=0, min = -1;

        int sectorsz=8;
        int groupsz=4096;
        int valsz=128;
        uint32_t blocksz=valsz*sectorsz;

        int nsector, ntoken, nspot, esector, etoken, espot;
        int tracker;


        for( sparse_list *x = this; x; x = x->next ) {
            if( x->size == 0 ) continue;


            if( cnt < 4 ) {
                if( kv ) {

                    // each token is 2048 bytes (8*128*2)
                    // tokens are in order of 4096 context

                    //n_embd_head_k, n_kv, n_head_kv,
                    // (128, 4096, 8)
                    nsector = x->offset/blocksz;
                    ntoken = x->offset/blocksz;
                    ntoken = (x->offset-nsector*blocksz)/valsz;
                    nspot = (x->offset-(nsector*blocksz+ntoken*valsz));
                    esector = (x->offset+x->size)/blocksz;
                    etoken = ((x->offset+x->size)-esector*blocksz)/valsz;
                    espot = ((x->offset+x->size)-(esector*blocksz+etoken*valsz));

                } else {

                    // each token is 2 bytes in 1024 places (2048 bytes total)
                    // from one place to the next is 4096 entries (8192 bytes)

                    nsector = x->offset/(2*groupsz*valsz);
                    tracker = nsector*2*groupsz*valsz;
                    nspot = (x->offset-tracker)/(2*valsz);
                    tracker += nspot*2*valsz;
                    ntoken = (x->offset-tracker)/2;

                    esector = (x->offset+x->size)/(2*groupsz*valsz);
                    tracker = esector*2*groupsz*valsz - x->offset;
                    espot = (x->size-tracker)/(2*valsz);
                    tracker = espot*2*valsz;
                    etoken = (x->size-tracker)/2;
                }
                std::string v;
                std::ostringstream vs;

                for( int z=0; z<x->size && z < 8; z+=2 ) {
                    vs << (float)( GGML_FP16_TO_FP32( *(ggml_fp16_t*)((void*)((char*)x->data+z)) ) ) << " ";
                }
                vs << "\n";
                v = vs.str();
                LLAMA_LOG_INFO("patch_report(%s[%d]) @(%d-%d) s:t+p (%d:%d+%d) - (%d:%d+%d)\n%s\n", kv?"key":"val", il, x->offset, x->size, nsector, ntoken, nspot, esector, etoken, espot, v.c_str());
            }


            if( x->offset < min || min < 0 ) min = x->offset;
            cnt++;
            sum += x->size;
            max = (max > x->offset+x->size) ? max : (x->offset+x->size);
        }

        if( max == 0 ) return;

        LLAMA_LOG_INFO("patch_report(%s[%d]): %d-%d (%d entries, %d total, %f average)\n", kv?"key":"val", il, min, max, cnt, sum, (float)sum/(float)cnt);
    }

    ~Sparse_list()
    {
        delete next;
        free(data);
    }
};

struct Sparse_patch {
    sparse_list *klayer[32];
    sparse_list *vlayer[32];

    Sparse_patch(bool prealloc=true)
    {
        if( prealloc ) {
            for( int i=0; i<32; i++ ) {
                klayer[i] = new sparse_list();
                vlayer[i] = new sparse_list();
            }
        }
    }
    void load(uint8_t *buf, int bufsz)
    {
        int layer_sz;
        uint8_t *ptr = buf;

        for( int i=0; i<32; i++ ) {
            memcpy( &layer_sz, ptr, sizeof(int) );
            klayer[i]->load( ptr+sizeof(int), layer_sz );
            ptr += layer_sz + sizeof(int);
            memcpy( &layer_sz, ptr, sizeof(int) );
            vlayer[i]->load( ptr+sizeof(int), layer_sz );
            ptr += layer_sz + sizeof(int);
        }
    }
    uint8_t *save( int *out_sz)
    {
        int sz=0, onesz;
        uint8_t *bufs[64];
        int sizes[64];

        for( int i=0; i<32; i++ ) {
            bufs[i] = klayer[i]->save(&sizes[i]);
            bufs[i+32] = vlayer[i]->save(&sizes[i+32]);
            sz += sizes[i] + sizes[i+32];
        }

        sz += 64*sizeof(int);

        uint8_t *tgt = (uint8_t*)malloc(sz);
        uint8_t *ptr=tgt;
        for( int i=0; i<32; i++ ) {
            memcpy( ptr, &sizes[i], sizeof(int) );
            memcpy( ptr+sizeof(int), bufs[i], sizes[i]);
            ptr += sizeof(int) + sizes[i];
            memcpy( ptr, &sizes[i+32], sizeof(int) );
            memcpy( ptr+sizeof(int), bufs[i+32], sizes[i+32]);
            ptr += sizeof(int) + sizes[i+32];
        }

        *out_sz = sz;
        return tgt;
    }

    sparse_patch *copy(void)
    {
        sparse_patch *tgt = new sparse_patch(false);
        for( int i=0; i<32; i++ ) {
            tgt->klayer[i] = klayer[i]->copy();
            tgt->vlayer[i] = vlayer[i]->copy();
        }
        return tgt;
    }

    sparse_patch *merge( sparse_patch *b, bool inplace=false )
    {
        sparse_patch *tgt = inplace ? this : this->copy();

        for( int i=0; i<32; i++ ) {
            tgt->klayer[i]->merge( b->klayer[i], true );
            tgt->vlayer[i]->merge( b->vlayer[i], true );
        }

        return tgt;
    }

    void readchanges( int il, bool kv, void *data, void *newbuf, int sz, bool commit_changes=false )
    {
        uint8_t *newdata = (uint8_t*)newbuf;
        uint8_t *buffer = (uint8_t*)data;

        int modoff=-1;
        sparse_list *tgt = kv ? klayer[il]:vlayer[il];
        int changes=0;

        for( int i=0; i<sz; i++ ) {
            if( buffer[i] != newdata[i] ) {
                changes++;
                if( modoff == -1 ) {
                    modoff = i;
                    continue;
                }
            } else {
                if( modoff != -1 ) {
                    tgt->append(modoff, i-modoff, newdata+modoff);
                    if( commit_changes ) memcpy(buffer+modoff, newdata+modoff, i-modoff);
                    modoff=-1;
                }
            }
        }
        if( modoff != -1 ) {
            tgt->append(modoff, sz-modoff, newdata+modoff);
            if( commit_changes ) memcpy(buffer+modoff, newdata+modoff, sz-modoff);
        }
        if( changes > 0 ) {
            //LLAMA_LOG_INFO("readchanges(): %d\n", changes);
        }
    }

    void apply( int il, bool kv, void *tgtbuf )
    {
        sparse_list *layer = kv ? klayer[il] : vlayer[il];
        layer->apply(tgtbuf);
    }

    void report()
    {
        for( int i=0; i<32; i++ ) {
            klayer[i]->report(true, i);
            vlayer[i]->report(false,i);
        }
    }

    ~Sparse_patch()
    {
        for( int i=0; i<32; i++ ) {
            delete klayer[i];
            delete vlayer[i];
        }
    }
};