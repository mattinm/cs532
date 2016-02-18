#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <unordered_map>
#include <vector>

#define READ_LEN    40
#define CHR_LEN     51

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
                 vector<string> &chr,
                 vector<string> &read )
{
    int current = ARG_NONE;
    int complete = ARG_NONE;

    for ( ; argc > 0; --argc, ++argv) {
        // first check for a new type
        if (strcmp(*argv, "--chr") == 0) {
            complete |= current;
            current = ARG_CHR;
            continue;
        } else if (strcmp(*argv, "--read") == 0) {
            complete |= current;
            current = ARG_READ;
            continue;
        }

        string s(*argv);
        if (current == ARG_NONE) {
            cerr << "Unknown argument: " << *argv << endl;
            chr.clear();
            read.clear();
            return false;
        } else if (current == ARG_CHR) {
            cout << "CHR  += '" << s << "'" << endl;
            chr.push_back(s);
        } else if (current == ARG_READ) {
            cout << "READ += '" << s << "'" << endl;
            read.push_back(s);
        } else {
            cerr << "Unknown parsing error" << endl;
            chr.clear();
            read.clear();
            return false;
        }
    }

    complete |= current;
    return (complete & (ARG_CHR | ARG_READ)) == (ARG_CHR | ARG_READ);
}

bool isValid(char c)
{
    return  c == 'N' || c == 'A' || c == 'C' || c == 'G' || c == 'T' ||
            c == 'n' || c == 'a' || c == 'c' || c == 'g' || c == 't';
}

bool nCount(char *s, int n)
{
    int n_count = 0;
    for (int i = 0; i < n; ++i)
        if (s[i] == 'N' || s[i] == 'n') n_count++;

    return n_count;
}

void print_usage()
{
    cout << "Usage: ./genome N_READS N_CHR --chr CHR_FILE [CHR_FILE ...] --read READ_FILE [READ_FILE ...]" << endl;
}

int kill()
{
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
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

        // consume the first arg
        --argc; ++argv;

        // read in the n-max for reads and chr
        if ((n_reads = strtol(*argv++, NULL, 10)) < 0) {
            print_usage();
            return kill();
        }
        if ((n_chr = strtol(*argv++, NULL, 10)) < 0) {
            print_usage();
            return kill();
        }

        // parse our file args
        if (!parse_args(argc-2, argv, chr_files, read_files)) {
            print_usage();
            return kill();
        }

        cout << "Read files: " << read_files.size() << endl;

        // read in our read files
        vector<char> reads;
        fstream fs;
        char buf[256];
        for (auto it = read_files.begin(); it != read_files.end(); ++it) {
            // open the file
            string filename = *it;
            cout << "Opening file: " << filename << endl;
            fs.open(filename, fstream::in);
            if (!fs.is_open()) {
                cerr << "Failed to open file: " << filename;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }

            // read in all our required lines
            while (1) {
                fs.getline(buf, 256);

                // terminate if not good read
                if (!fs.good()) break;

                // insert the line (WITH '\0') to our reads vector
                if (isValid(*buf) && nCount(buf, READ_LEN) < n_reads)
                    reads.insert(reads.end(), buf, buf+READ_LEN+1);
            }

            // close the file and go to the next one
            fs.close();
        }

        // print out our reads
        cout << "Reads length: " << reads.size() << endl;
    }

    if (rank == 0) {
        // go through the chr files
        for (auto it = chr_files.begin(); it != chr_files.end(); ++it) {
            string filename = *it;
            char buf[CHR_LEN];

            // skip the first line
            fs.getline(buf, CHR_LEN);
            if (!fs.good()) continue;

            // read in each line
            while (1) {
                fs.getline(buf, CHR_LEN);
                if (!fs.good()) break;
            }
        }
    }

    MPI_Finalize();
    return 0; 
}
