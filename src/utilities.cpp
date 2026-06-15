#include "utilities.h"
#include "header.h"

#include <fstream> // fscanf, fopen, ofstream
#include <sstream>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/**
 *
 * @param p_Labels
 * @param p_sOutputFile
 */
void outputLabels(const IVector & p_Labels, const string& p_sOutputFile)
{
//	cout << "Outputing File..." << endl;
    ofstream myfile(p_sOutputFile);

    //cout << p_matKNN << endl;

    for (auto const& i : p_Labels)
    {
        myfile << i << '\n';
    }

    myfile.close();
//	cout << "Done" << endl;
}

/**
 * Load data (each line is a point) into MatrixXf of size D x N format
 * Check the supporting distance and apply normalization (cosine, chi2, JS)
 *
 * @param dataset
 * @param distance
 * @param numPoints
 * @param numDim
 * @param MATRIX_X
 */
void loadtxtData(const string& dataset, const string& distance, int numPoints, int numDim, RowMajorMatrixXf & MATRIX_X) {

    FILE *f = fopen(dataset.c_str(), "r");
    if (!f) {
        cerr << "Error: Data file does not exist !" << endl;
        exit(1);
    }

    // Important: If use a temporary vector to store data, it doubles the memory
    // MATRIX_X = MatrixXf::Zero(numDim, numPoints); // default is col-major
    MATRIX_X = RowMajorMatrixXf::Zero(numPoints, numDim); // row-wise

    // Each line is a vector of d dimensions
    for (int n = 0; n < numPoints; ++n) {
        for (int d = 0; d < numDim; ++d) {
            // fscanf(f, "%f", &MATRIX_X(d, n)); // col-major
            fscanf(f, "%f", &MATRIX_X(n, d)); // row-major
        }
    }

    cout << "Finish reading data" << endl;

    //        MATRIX_X.transpose();
    //        cout << "X has " << MATRIX_X.rows() << " rows and " << MATRIX_X.cols() << " cols " << endl;

    /**
    Print the first col (1 x N)
    Print some of the first elements of the MATRIX_X to see that these elements are on consecutive memory cell.
    **/
    //        cout << MATRIX_X.row(0) << endl << endl;
    //        cout << "In memory (col-major):" << endl;
    //        for (n = 0; n < 10; n++)
    //            cout << *(MATRIX_X.data() + n) << "  ";
    //        cout << endl << endl;

    cout << "Now checking the condition of data given the distance." << endl;
    transformData(MATRIX_X, distance);
}

void loadbinData(const string& dataset, const string& distance, int numPoints, int numDim, RowMajorMatrixXf & MATRIX_X) {

    // Open file
    int fd = open(dataset.c_str(), O_RDONLY);
    if (fd < 0) {
        perror("open");
        exit(1);
    }

    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("fstat");
        exit(1);
    }

    size_t filesize = sb.st_size;
    size_t total_rows = filesize / (numDim * sizeof(float));

    std::cout << "Total rows = " << total_rows << std::endl;

    // Map the file into memory
    void* mapped = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    close(fd); // fd no longer needed

    if ((size_t)numPoints > total_rows) {
        std::cerr << "Error: numPoints exceeds the number of rows in the file." << std::endl;
        munmap(mapped, filesize);
        exit(1);
    }

    // Important: If use a temporary vector to store data, it doubles the memory
    // MATRIX_X = MatrixXf::Zero(numDim, numPoints); // default is col-major
    MATRIX_X = RowMajorMatrixXf::Zero(numPoints, numDim);

    // Interpret data as float array
    float* data = reinterpret_cast<float*>(mapped);

    // Each line is a vector of d dimensions
    for (int n = 0; n < numPoints; ++n) {
        for (int d = 0; d < numDim; ++d) {
            // MATRIX_X(d, n) = data[n * numDim + d]; // col-major
            MATRIX_X(n, d) = data[n * numDim + d]; // row-major
        }
    }


    // Unmap when done
    munmap(mapped, filesize);

    cout << "Finish reading data" << endl;

    //        MATRIX_X.transpose();
    //        cout << "X has " << MATRIX_X.rows() << " rows and " << MATRIX_X.cols() << " cols " << endl;

    /**
    Print the first col (1 x N)
    Print some of the first elements of the MATRIX_X to see that these elements are on consecutive memory cell.
    **/
    //        cout << MATRIX_X.row(0) << endl << endl;
    //        cout << "In memory (col-major):" << endl;
    //        for (n = 0; n < 10; n++)
    //            cout << *(MATRIX_X.data() + n) << "  ";
    //        cout << endl << endl;

    cout << "Now checking the condition of data given the distance." << endl;
    transformData(MATRIX_X, distance);
}

/**
 * Normalize data to support the distance (only needed for Cosine, Chi2, JS)
 * @param MATRIX_X
 * @param distance
 */
void transformData(RowMajorMatrixXf & MATRIX_X, const string& distance)
{
    // Check support distance
    // Doing cross-check for normalize points with cosine, and non-negative values for Chi2 and JS
    int numPoints = MATRIX_X.rows();

    if (distance == "Cosine")
    {
#pragma omp parallel for
        for (int n = 0; n < numPoints; ++n)
            MATRIX_X.row(n) /= MATRIX_X.row(n).norm(); // or MATRIX_X.rowwise().normalize() inplace but not multi-threading

//        cout << MATRIX_X.row(0).norm() << endl;
//        cout << MATRIX_X.row(10).norm() << endl;
//        cout << MATRIX_X.row(100).norm() << endl;
    }
    else if ((distance == "Chi2") || (distance == "JS"))
    {
        // Ensure non-negative
        if (MATRIX_X.minCoeff() < 0)
        {
            cerr << "Error: X is not non-negative !" << endl;
            exit(1);
        }
        else // normalize to have sum = 1
        {
            // Get colwise.sum is a row array, need to transpose() to make it col array
#pragma omp parallel for
            for (int n = 0; n < numPoints; ++n)
            {
                float fSum = MATRIX_X.row(n).sum();
                if (fSum <= 0)
                {
                    cerr << "Error: There is an zero point !" << endl;
                    exit(1);
                }
                MATRIX_X.row(n) /= fSum;
            }

            // Test
//            cout << MATRIX_X.row(0).sum() << endl;
//            cout << MATRIX_X.row(10).sum() << endl;
//            cout << MATRIX_X.row(100).sum() << endl;
        }
    }

}

