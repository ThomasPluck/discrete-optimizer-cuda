#include <iostream>
#include <vector>
#include <fstream>
#include "structs.h"

#define BATCH 32

#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_SIZE 784
#define MNIST_DATA_LENGTH 60000
#define MNIST_DATA_THRESHOLD 50
#define MNIST_NUM_CLASSES 10

using namespace std;

#pragma region MNIST

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

uchar * ReadMNISTImages(string path)
{
    ifstream file (path,std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        uchar output[number_of_images*n_rows*n_cols];
        unsigned char temp=0;

        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    file.read((char*)&temp,sizeof(temp));
                    output[(i*n_rows+r)*n_cols + c] = temp;
                }
            }
        }

        return output;
    }
}

uchar * ReadMNISTLabels(string path)
{
    ifstream file (path,std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_labels=0;

        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels= reverseInt(number_of_labels);

        uchar output[number_of_labels];
        unsigned char temp=0;

        for(int i=0;i<number_of_labels;++i)
        {
            file.read((char*)&temp,sizeof(temp));
            output[i] = temp;
        }

        return output;
    }
}

#pragma endregion


Host_Matrix PackHostMatrix(uchar * DataToPack, int bit_width, int bit_height, int byte_threshold) {

    Host_Matrix output(bit_width,bit_height);

    uint8_t bits = 0;

    // Binarize the matrix in row major format
    for (int i = 0; i < PAD8(bit_height); i++) {
    for (int j = 0, k = 0; j < PAD128(bit_width); j++, k++) {

        k %= 8;
        // If you've reached of the byte and you're within in bounds, commit it.
        if (k == 0 && j != 0) {
            output(j%PAD128(bit_width),i) = bits;
            bits = 0;
        }

        // If you're within bounds, binarize and add to final data output.
        if (i < bit_height && j < bit_width) {
            bits += (DataToPack[i*bit_width+j] > byte_threshold ? (uchar) 128 : (uchar) 0) >> k;
        } else {
            continue;
        }
        
    }}

    output.upload();

    return output;
}