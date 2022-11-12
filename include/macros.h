//how many bits steps to go
#define STEP4(X) (((X)+3)>>2) 
#define STEP8(X) (((X)+7)>>3) 
#define STEP16(X) (((X)+15)>>4) 
#define STEP32(X) (((X)+31)>>5) 
#define STEP64(X) (((X)+63)>>6) 
#define STEP128(X) (((X)+127)>>7)
#define STEP256(X) (((X)+255)>>8)

//total bits covers after padding
#define PAD4(X) (STEP4(X)<<2)
#define PAD8(X) (STEP8(X)<<3)
#define PAD16(X) (STEP16(X)<<4)
#define PAD32(X) (STEP32(X)<<5)
#define PAD64(X) (STEP64(X)<<6)
#define PAD128(X) (STEP128(X)<<7)

//get bytesize of matrix from bit dimensions
#define MATRIXSIZE(X,Y) (PAD128(X)/8*PAD8(Y))

//get lane id
#define GET_LANEID unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid)); 
//get warp id
#define GET_WARPID unsigned warpid; asm("mov.u32 %0, %%warpid;":"=r"(warpid)); 

//model parameters
#define WEIGHT_THRESHOLD 256
#define BIAS_THRESHOLD 4
#define DEFAULT_SEED 694201337