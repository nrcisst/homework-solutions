#include <fstream>
#include <iostream>
#include <cctype>
#include<string>
#include "ArgumentManager.h"

using namespace std;

int Partition(long arr[], int lowIn, int highIn){
    long  pivot= arr[highIn];
    int i= lowIn -1;
    
    for(int j=lowIn; j<highIn; j++){
        if(arr[j] < pivot){
            i++;
            long temp= arr[i];
            arr[i]= arr[j];
            arr[j] = temp;
        }
    }
    long temp= arr[i + 1];
    arr[i+1] = arr[highIn];
    arr[highIn]=temp;
    return i+1;
}

long Quick(long arr[], int lowIn, int highIn, int k){
    if(lowIn <= highIn){
        int pivot = Partition(arr , lowIn, highIn);
        if (pivot == k){
            return arr[pivot];
        }else if (pivot < k){
            return Quick(arr, pivot + 1, highIn, k);
        }else{
            return Quick(arr, lowIn, pivot - 1, k);
        }
    }
    return 0;
}


int main(int argc, char  * argv[]) {
    //ArgumentManager am(argc, argv);
    string in = argv[1];//am.get("input");
    string output = argv[2];//am.get("output");
    
    ifstream input(in);
    ofstream out(output);
    if(!input){
        cout<<"Error: Unable to open file."<<endl;
        return 0;
    }
    
    int numValues;
    
    input>>numValues;
    long *arr = new long[numValues];
    
    for (int i=0; i< numValues; i++){
        input>>arr[i];
    }
    
    for (int i=0; i< numValues; i++){
        cout<<arr[i]<<endl;
    }
    
    int firstQ= numValues / 4;
    int secondQ = 3 * (numValues) / 4;
    int median = (numValues - 1)/2;
    
    cout<< "FirstQ: "<<Quick(arr, 0, numValues - 1, firstQ)<<endl;
    cout<<"Median: "<< Quick(arr, 0, numValues - 1, median)<<endl;
    cout<< "SecondQ: "<<Quick(arr, 0, numValues - 1, secondQ)<<endl;
    
    delete[] arr;
    return 0;
}
