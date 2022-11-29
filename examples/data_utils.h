#include <iostream>
#include <vector>
#include <fstream>
#include "structs.h"

using namespace std;

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void ReadMNIST(string ImagePath, int NumberOfImages, int DataOfAnImage, vector<vector<unsigned char>> &arr)
{
    arr.resize(NumberOfImages, vector<unsigned char>(DataOfAnImage));
    ifstream file (ImagePath,ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= temp;
                }
            }
        }
    }
}

uchar Bools2Char(bool bits[8]) {
    char c = 0;
    for (int i = 0; i < 8; i++) {
        c += (bits[i] >> i);
    }
    return c;
}

Host_Matrix ThresholdAndPack(vector<vector<uchar>> DataToPack, int Threshold) {
    
    uchar output[DataToPack.size(),DataToPack[0].size()/8+1];

    char bits = 0;
    int count = 0;
    // Binarize the matrix
    for (int i = 0; i < DataToPack.size(); i++) {
    for (int j = 0, k = 0; j < DataToPack[0].size(); j++, k++) {

        if (k == 0 && (j != 0 || i != 0)) {
            output[i*DataToPack.size()+(j/8)-1] = bits;
            bits = 0;
        }

        bits += (DataToPack[i][j] > Threshold ? (char) 128 : (char) 0) >> k;
        k %= 8;
    }}
    
    // Pack into Li's format
    Host_Matrix matrix_output;
    matrix_output.load(output,DataToPack[0].size(),DataToPack.size());

    return matrix_output;
}