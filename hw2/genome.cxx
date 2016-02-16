#include <fstream>
#include <iostream>
#include <mpi.h>
#include <unordered_map>

#define ARG_NONE    0
#define ARG_CHR     1
#define ARG_READ    2

using namespace std;

/** Parses the command line arguments and returns success or failure.
 *
 * @param argc the standard argc
 * @param argv the standard argv
 * @param chr reference to the vector of chr filenames
 * @param read reference to the vector of read filenames
 */
bool parse_args( int argc, char **argv,
                 std::vector<std::string> &chr,
                 std::vector<std::string> &read )
{
    int current = ARG_NONE;
    int complete = ARG_NONE;

    while (argc--) {
        // first check for a new type
        if (strcmp(*argv, "--chr")) {
            complete |= current;
            current = ARG_CHR;
            continue;
        } else if (strcmp(*argv, "--read")) {
            complete |= current;
            current = ARG_READ;
            continue;
        }

        switch (current) {
        case ARG_NONE:
            cout << "Unknown argument: " << *argv << endl;
            chr.clear();
            read.clear();
            return false;
        
        case ARG_CHR:
            chr.push_back(new std::string(*argv));
            break;

        case ARG_READ:
            read.push_back(new std::string(*argv));
            break;

        default:
            cout << "Unknown parsing error" << endl;
            chr.clear();
            read.clear();
            return false;
        }
    }

    complete |= current;
    return complete & ARG_CHR & ARG_READ;
}

int main(int argc, char **argv)
{
    // initialize MPI
    int comm_sz, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n_reads = 0;    // n's allowed in reads
    int n_chr = 0;      // n's allowed in chrs

    // parse our arguments
    if (rank == 0) {
        vector<string> chr_files;
        vector<string> read_files;

        if (!parse_args(argc, argv, &chr, &read)) {
            cout << "Usage: ./genome --chr CHR_FILE [CHR_FILE ...] --read READ_FILE [READ_FILE ...]" << endl;

            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        // read in our read files
        vector<char> reads;
        string filename;
        fstream fs;
        char buf[256];
        for (filename : read) {
            // open the file
            fs.open(filename, fstream::in);
            if (!fs.is_open()) {
                cout << "Failed to open file: " << filename;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }

            // read in all our required lines
            while (1) {
                fs.getline(buf, 256);
                if (!fs.good()) break;


            }

            fs.close();
        }
    }

    unordered_map<string, vector<int, int>> counts;

    /* read in all the read files */
    if (rank == 0) {
        string read;
        
    }

    return 0; 
}
