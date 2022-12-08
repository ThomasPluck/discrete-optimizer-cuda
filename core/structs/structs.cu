#include"structs.h"
#include"util.cuh"

#pragma region Chunk

    //constructor
    __device__ Chunk::Chunk(uchar* _data){
        #pragma unroll
        for (int i = 0; i < BLOCK_HEIGHT; i++)
        {
            data[i] = *(_data+i);
        }
    }

    // Henry Warren's Tranpose32 distilled to Transpose8
    __device__ void Chunk::transpose(){
        int8_t j, k;
        uchar m, t;

        for (j = 4, m = 0x0F; j; j >>= 1, m ^= m << j) {

            for (k = 0; k < 8; k = ((k | j) + 1) & ~j) {

                t = (data[k] ^ (data[k | j] >> j)) & m;

                data[k] ^= t;

                data[k | j] ^= (t << j);
            }
        }
    }

    //invert data
    __device__ void Chunk::NOT(){
        *halves[0] = ~*halves[0];
        *halves[1] = ~*halves[1];
    }

    //AND two chunks together
    __device__ Chunk ChunkAND(Chunk LHS, Chunk RHS){
        union{uchar data[8]; uin32 halves[2];} p0;
        p0.halves[0] = *LHS.halves[0] & *RHS.halves[0];
        p0.halves[1] = *LHS.halves[1] & *RHS.halves[1];

        return Chunk(p0.data);
    }

    // Quick bit-hack to count bits in a byte
    __device__ uchar BytePOPC(uchar n){
        uchar a = n;
        n = ((n & 0xAA) >> 1) + (n & 0x55);
        n = ((n & 0xCC) >> 2) + (n & 0x33);  
        n = ((n & 0xF0) >> 4) + (n & 0x0F);
        return n;
    }

    //Counts number of ones in chunk
    __device__ void ChunkPOPC(Chunk chunk, uchar* out){

        for (uchar i = 0; i < 8; i++){
            out[i] = BytePOPC(chunk.data[i]);
        }

    }


#pragma endregion

#pragma region Device_Matrix

    //constructor
    Device_Matrix::Device_Matrix(int block_width, int block_height, int bit_width, int bit_height, uchar* _data) : Device_Data<uchar>(_data){
        block_dims[0] = block_width;
        block_dims[1] = block_height;
        bit_dims[0] = bit_width;
        bit_dims[1] = bit_height;
        element_dims[0] = bit_width / CHAR_BIT;
        element_dims[1] = bit_height;
    }

    // Gets 8-by-8 bit chunk from the larger bit matrix
    __device__ Chunk Device_Matrix::get_chunk(uint eight_x, uint eight_y){

        uint block_x = eight_x / BLOCK_WIDTH;
        uint block_y = eight_y;
        uint internal_x = eight_x % BLOCK_WIDTH;

        uchar out_bytes[8];
        #pragma unroll
        for(int internal_y = 0; internal_y < BLOCK_HEIGHT; internal_y++){
            out_bytes[internal_y] = (*this)(block_x, block_y, internal_x, internal_y);
        }

        return Chunk(out_bytes);
    }

    // Sets 8-by-8 bit chunk into the larger bit matrix
    __device__ void Device_Matrix::set_chunk(uint eight_x, uint eight_y, Chunk chunk){
        uint block_x = eight_x / BLOCK_WIDTH;
        uint block_y = eight_y;
        uint internal_x = eight_x % BLOCK_WIDTH;

        #pragma unroll
        for(int internal_y = 0; internal_y < BLOCK_HEIGHT; internal_y++){
            (*this)(block_x, block_y, internal_x, internal_y) = chunk.data[internal_y];
        }
    }

    // Sets 8-by-8 chunk via two 32-bit half chunks
    __device__ void Device_Matrix::set_half_chunks(int eight_x, int eight_y, uchar* half_1, uchar* half_2){
        uint block_x = eight_x / BLOCK_WIDTH;
        uint block_y = eight_y;
        uint internal_x = eight_x % BLOCK_WIDTH;

        #pragma unroll
        for(int internal_y = 0; internal_y < CHUNK_HEIGHT/2; internal_y++){
            (*this)(block_x, block_y, internal_x, internal_y) = half_1[internal_y];
        }
        
        #pragma unroll
        for(int internal_y = 0; internal_y < CHUNK_HEIGHT/2; internal_y++){
            (*this)(block_x, block_y, internal_x, internal_y+4) = half_2[internal_y];
        }
    }

    // __device__ __host__ Device_Matrix Device_Matrix::transpose(){
    //     Launch::kernel_2d(dims[0], dims[1]);
    //     Transpose<<<Launch::num_blocks, Launch::threads_per_block>>>((*this),(*this).bit_width(),(this*).bit_height());
    //     SYNC_KERNEL(Transpose);
    // }


#pragma endregion

#pragma region Host_Matrix

    //default constructor
    Host_Matrix::Host_Matrix() : Host_Data<uchar>(){}

    //real constructor
    Host_Matrix::Host_Matrix(int bit_width, int bit_height) : Host_Data<uchar>(MATRIXSIZE(bit_width,bit_height)){
        block_dims[0] = STEP128(bit_width);
        block_dims[1] = STEP8(bit_height);
        bit_dims[0] = bit_width;
        bit_dims[1] = bit_height;
        element_dims[0] = bit_width / CHAR_BIT;
        element_dims[1] = bit_height;
    }

    Host_Matrix::~Host_Matrix(){}

    Host_Matrix::Host_Matrix(const Host_Matrix& input) : Host_Data<uchar>(input){
        block_dims[0] = input.block_dims[0];
        block_dims[1] = input.block_dims[1];

        bit_dims[0] = input.bit_dims[0];
        bit_dims[1] = input.bit_dims[1];

        element_dims[0] = input.element_dims[0];
        element_dims[1] = input.element_dims[1];
    }

    void Host_Matrix::operator=(const Host_Matrix& input){
        
        Host_Data::operator=((Host_Data) input);

        block_dims[0] = input.block_dims[0];
        block_dims[1] = input.block_dims[1];

        bit_dims[0] = input.bit_dims[0];
        bit_dims[1] = input.bit_dims[1];

        element_dims[0] = input.element_dims[0];
        element_dims[1] = input.element_dims[1];
    }

    // Load Host_Matrix with row-major bit data in a single uchar array
    void Host_Matrix::load(uchar* data, int bit_width_, int bit_height_){
 
        num_blocks_width() = STEP128(bit_width_);
        num_blocks_height() = STEP8(bit_height_);
        bit_width() = bit_width_;
        bit_height() = bit_height_;

        for(int i = 0; i < bytesize; i++){
            host_data[i] = data[i];
        }

        upload();
    }

    Host_Matrix::operator Device_Matrix() {
        return Device_Matrix(num_blocks_width(), num_blocks_height(), bit_width(), bit_height(), device_data);
    }


#pragma endregion











