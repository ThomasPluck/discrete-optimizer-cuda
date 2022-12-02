#include <iostream>
#include <vector>
#include <fstream>
#include "structs.h"

using namespace std;

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

uchar** ReadMNIST(string path)
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

        uchar ** output;
        output = new uchar*[number_of_images];
        for (int i = 0; i < n_rows * n_cols; i++) {
            output[i] = new uchar[n_rows * n_cols];
        }

        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    output[i][r*n_cols + c] = temp;
                }
            }
        }

        return output;
    }
}

uchar Bools2Char(bool bits[8]) {
    char c = 0;
    for (int i = 0; i < 8; i++) {
        c += (bits[i] >> i);
    }
    return c;
}

Host_Matrix ThresholdAndPack(uchar ** DataToPack, int Threshold) {
    
    uchar output[PAD8(sizeof(DataToPack))*PAD128(sizeof(DataToPack[0]))];

    uint8_t bits = 0;

    // Binarize the matrix in row major format
    for (int i = 0; i < PAD8(sizeof(DataToPack)); i++) {
    for (int j = 0, k = 0; j < PAD128(sizeof(DataToPack[0])); j++, k++) {

        k %= 8;

        if (k == 0 && (j != 0 || i != 0)) {
            output[i*PAD128(sizeof(DataToPack[0]))+(j/8)] = (uchar) bits;
            bits = 0;
        }

        try{
            bits += (DataToPack[i][j] > Threshold ? (char) 128 : (char) 0) >> k;
        } catch (...) {
            continue;
        }
    }}

    // Flatten output for Host_Matrix Packing
    uchar flattened[sizeof(DataToPack)*sizeof(DataToPack[0])];
    for (int i = 0; i < sizeof(DataToPack); i++) {
    for (int j = 0, k = 0; j < sizeof(DataToPack[0]); j++) {
        flattened[i*sizeof(DataToPack[0]) + j] = output[i][j];
    }}
    
    // Pack into Li's format
    Host_Matrix matrix_output;
    matrix_output.load(flattened,sizeof(DataToPack[0]),sizeof(DataToPack));

    return matrix_output;
}