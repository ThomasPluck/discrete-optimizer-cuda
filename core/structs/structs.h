#pragma once

#include"standard_includes.h"
#include"cuda_includes.h"

#include"macros.h"
#include"launch.h"


typedef unsigned int uint;
typedef unsigned int uin32;
typedef unsigned char uchar;
typedef unsigned char uin8;

// define magic number macros

// Blocks are 8-by-128 bit subarrays
const uint block_height = 8;
const uint block_width = 16;
const uint block_size = block_height * block_width;

// Chunks are 8-by-8 bit subarrays
const uint chunk_width = 1;
const uint chunk_height = 8;
const uint chunk_size = 8;

// Half Chunks are 4-by-8 bit subarrays
const uint half_chunk_height = 4;

template<typename Type>
struct Device_Data{

    Type* data;

    //constructor
    Device_Data(Type* _data){
        data = _data;
    }

    //data access
    __device__ inline Type& operator[](int index){
        return data[index];
    }
};

template <typename Type>
struct Host_Data{

    Type* host_data;
    Type* device_data;

    int bytesize;

    bool initialized = false;

    //default constructor
    Host_Data(){}

    //real constructor
    Host_Data(int _bytesize, int constant = 0){
        bytesize = _bytesize;
        allocate();
        fill(constant);
        initialized = true;
    }

    //destructor
    ~Host_Data(){
        if(initialized){
            delete host_data;
            CUDA_SAFE_CALL(cudaFree(device_data));
        }
    }

    //copy constructor
    Host_Data(const Host_Data<Type>& input){
        bytesize = input.bytesize;
        allocate();
        load(input.host_data);
        initialized = true;
    }

    //copy assignment operator
    void operator= (const Host_Data<Type>& input){
        if(initialized){
            delete host_data;
            cudaFree(device_data);
        }

        bytesize = input.bytesize;
        allocate();
        load(input.host_data);
        initialized = true;
    }

    //allocate memory
    void allocate(){
        host_data = new Type[bytesize];
        cudaMalloc((void**)&device_data, bytesize);
    }

    //fill memory with random values
    void fill_random(int seed= DEFAULT_SEED){
        srand(seed);
        for(int i = 0 ; i < bytesize ; i++){
            host_data[i] = rand();
        }
        upload();
    }


    //fill memory with a constant value
    void fill(int value = 0){
        for(int i = 0; i < bytesize; i++){
            host_data[i] = value;
        }
        upload();
    }

    //load host data from pointer
    void load(Type* input){
        for(int i = 0; i < bytesize; i++){
            host_data[i] = input[i];
        }
        upload();
    }

    //upload memory to device from host
    void upload(){
        cudaMemcpy(device_data, host_data, bytesize, cudaMemcpyHostToDevice);
    }

    //download memory to host from device
    void download(){
        cudaMemcpy(host_data, device_data, bytesize, cudaMemcpyDeviceToHost);
    }

    //indexed access of host data
    __host__ Type& operator[](int index){
        return host_data[index];
    }

    //typecast to Device_Data
    operator Device_Data<Type>(){
        return Device_Data<Type>(device_data);
    }

};

/*!
* === The Chunk Class ===
* 
* The chunk class exists to easily store and manipulate 8-by-8 bit chunks that represent 8-by-8 bit squares
* in the larger Matrix classes own underlying 2D bit array.
*
* It is designed to be specifically instantiated by the `get_chunk` method in the Matrix class.
*
* It also contains a method to transpose itself at the bit-level which is functionality necessary for TCBNN++
*/

struct Chunk {

    uchar data[8] = {0};
    uin32* halves[2] = {(uin32*)&data[0],
                        (uin32*)&data[half_chunk_height]};

    //constructor
    __device__ Chunk(uchar* _data);

    // Henry Warren's Tranpose32 distilled to Transpose8
    __device__ void transpose();

    //invert data
    __device__ void NOT();
};

//AND two chunks together
__device__ Chunk ChunkAND(Chunk LHS, Chunk RHS);

// Quick bit-hack to count bits in a byte
__device__ uchar ByteCount(uchar n);

// Counts number of ones in chunk
__device__ void ChunkPOPC(Chunk chunk, uchar* out);

/*! ==== The Matrix Class ====
*
* The Matrix class is written expressly to easily handle and manipulate bit-arrays stored in Li's format
*
* Li's Format:
*
* Li's format packs 2D bit arrays (we all this bit array the Matrix) such that 8-by-128 2D bit sub-arrays (8-by-16 2D byte arrays or "blocks")
* are arranged in row-major format contiguously in memory. The reason for this, is that BMMA calls operate optimally on 
* 8-by-128 bit blocks and so this format actually limits the amount of address translation required to perform BMMA (see Ang Li's TCBNN).
* 
* How this is done in practice, is that 8-by-128 bit blocks are first stored in row major format, where adding one to the final pointer
* points to the first element of the next 8-by-128 bit block in the "block row" of 8-by-128 bit subarrays that tile the original matrix.
* And naturally, the final pointer of the block row incremented, points to the first element of the first block in the next block row.
* 
* This creates a nested row-major 4D array, where on the byte level we increment across the interior of the block and on the block level
* we increment over block-rows. 
*
* We manage this with the use of the Matrix class, which contains the original byte-array and an operator to navigate using the 4D coordinate
* system that we have decided to call Li coordinates. As well as GET/SET methods for 8-by-8 bit chunks which represent 8-by-8 bit squares in
* the original 2D bit array we call Matrix.
*/

struct Device_Matrix : public Device_Data<uchar>{

    uint block_dims[2] = {1,1};
    uint bit_dims[2] = {1,1};
    uint element_dims[2] = {1,1};

    __device__ __host__ inline uint& num_blocks_width() {return block_dims[0];}
    __device__ __host__ inline uint& num_blocks_height() {return block_dims[1];}
    __device__ __host__ inline uint& bit_width() {return bit_dims[0];}
    __device__ __host__ inline uint& bit_height() {return bit_dims[1];}

    Device_Matrix(int block_width, int block_height, int bit_width, int bit_height, uchar* _data);

    /*!
    * ==== Understanding Li Coordinates ====
    * 
    * Reminder:
    * 
    * Consider standard coordinates of a byte array with byte-width W to retrieve the pointer of
    * the byte located at (x,y) you use the arithmetic:
    * 
    * p = x + W * y
    * 
    * Li Coordinates:
    * 
    * To manage the madness, we created a system we call Li's coordinates, bytes are first ordered by their internal block dimensions
    * internal_x and internal_y which are always 0 < ix < 16, 0 < iy < 8 and then are ordered by their block dimensions block_x and block_y
    * which are arbitrarily large.
    * 
    * Li's coordinates parameterize byte-arrays with dimensions that are multiples of the 8-by-16 byte chunks.
    * 
    * To calculate Li coordinates from standard coordinates, the following formulae are used:
    * 
    * block_x = x / 16
    * block_y = y / 8
    * internal_x = x % 16
    * internal_y = x % 8
    * 
    * To then derive the pointer to a byte in a matrix with W horizontal chunks (ie. chunk width) parameterized
    * by Li coordinates the following formula is used:
    * 
    * p = 16*8*bx + W*16*8*by + ix + 16*iy (linear access) = ((by*W+bx)*block_height+iy)*block_width+ix (vectorized)
    */

    // Access bytes using Li coordinates
    __device__ inline uchar& operator()(uint block_x, uint block_y, uint internal_x, uint internal_y ){
        return data[((block_y * num_blocks_width() + block_x ) * block_height + internal_y ) * block_width + internal_x];
    }

    // Access bytes using vectorized coordinates
    __device__ inline uchar& operator()(uint byte_x, uint byte_y){
        return data[(byte_x + block_width * byte_y) + (block_size*(byte_x/block_width)) + (block_size*num_blocks_width()*(byte_y/block_height))];
    }

    // Access a byte array representing a row via a single index
    __device__ inline uchar * operator()(uint row){

        const int row_bytes = num_blocks_width()*block_width;

        uchar * output = new uchar[row_bytes];

        for (int i; i < row_bytes; i++) {
            output[i] = (*this)(i,row);
        }

        return output;

    }

    // Access pointer using Li coordinates
    __device__ inline const void * ptr(uint block_x, uint block_y, uint internal_x, uint internal_y ){
        return (void *) data + (((block_y * num_blocks_width() + block_x ) * block_height + internal_y ) * block_width + internal_x);
    }

    // Access pointer using vectorized coordinates
    __device__ inline const void * ptr(uint byte_x, uint byte_y){
        return (void *) data + ((byte_x + block_width * byte_y) + (block_size*(byte_x/block_width)) + (block_size*num_blocks_width()*(byte_y/block_height)));
    }

    // Access a pointer representing a row via a single index
    __device__ inline const void* ptr(uint row){

        return ptr(0,row);

    }

    /*!
    * ==== Understanding "8 Chunk" Coordinates ====
    *
    * 8 Chunk Coordinates are used to navigate matrices packed using Li coordinates to easily access
    * square 8-by-8 bit-arrays (ie. 8-byte arrays) which simplifies computation of bit-level manipulation
    * needed to make TCBNN++ function.
    * 
    * We can imagine a bit-array as being cut up into a grid of 8-by-8 bit-chunks that are square. To simply access and manipulate
    * these - we can create a coordinate system on the 8-by-8 bit-chunk grid which is what the below operators allow:
    * 
    * To retrieve the pointer with 8-chunk coordinates (eight_x,eight_y) with 128-chunk width W in a matrix packed using
    * Li coordinates, we can use the formula below:
    * 
    * p = eight_x % 16 + 8*16*eight_x/16 + eight_y*16*8*W
    * 
    * And in Li coordinates explicitly:
    * 
    * block_x = eight_x / 16
    * block_y = eight_y
    * internal_x = eight_x % 16
    * internal_y = 0
    */

    // Gets 8-by-8 bit chunk from the larger bit Matrix
    __device__ Chunk get_chunk (uint eight_x, uint eight_y);

    // Sets 8-by-8 bit chunk into the larger bit matrix
    __device__ void set_chunk (uint eight_x, uint eight_y, Chunk chunk);
    
    // Sets 8-by-8 chunk via two 32-bit half chunks
    __device__ void set_half_chunks(int eight_x, int eight_y, uchar* half_1, uchar* half_2);

    // Returns transposed matrix
    __device__ Device_Matrix transpose();

};

struct Host_Matrix : public Host_Data<uchar>{

    //dims in terms of blocks
    uint block_dims[2] = {1,1};
    uint bit_dims[2] = {1,1};
    uint element_dims[2] = {1,1};

    __host__ inline uint& num_blocks_width() {return block_dims[0];}
    __host__ inline uint& num_blocks_height() {return block_dims[1];}
    __host__ inline uint& bit_width() {return bit_dims[0];}
    __host__ inline uint& bit_height() {return bit_dims[1];}

    //default constructor
    Host_Matrix();

    Host_Matrix(int bit_width, int bit_height);

    ~Host_Matrix();

    Host_Matrix(const Host_Matrix& input);

    void operator=(const Host_Matrix& input);

    void load(uchar* data, int _bit_width, int _bit_height);

    operator Device_Matrix();

};





