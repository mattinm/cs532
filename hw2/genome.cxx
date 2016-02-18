#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <unordered_map>
#include <vector>

#define READ_LEN    40
#define CHR_LEN     51
#define CHR_COUNT   100
#define BUF_LEN     256

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

/** Converts a string to uppercase.
 * @param s the string to convert
 * @param n the length of the string
 */
void stringToUpper(char *s, int n)
{
    while (n--)
        s[n] = toupper(s[n]);
}

/** Checks to see if the character is valid.
 * @param c the character to check
 */
bool isValid(char c)
{
    return  c == 'N' || c == 'A' || c == 'C' || c == 'G' || c == 'T';
}

/** Counts all of the N's in the given string.
 * @param s the string to count
 * @param n the length to count
 */
int nCount(char *s, int n)
{
    int n_count = 0;
    for (int i = 0; i < n; ++i)
        if (s[i] == 'N') n_count++;

    return n_count;
}

/** Prints out the usage of the program. */
void print_usage()
{
    cout << "Usage: ./genome N_READS N_CHR --chr CHR_FILE [CHR_FILE ...] --read READ_FILE [READ_FILE ...]" << endl;
    cout << "============================================================================================" << endl;
    cout << "N_READS       max number of N's in read files" << endl;
    cout << "CHR_READS     max number of N's in chr files" << endl;
    cout << "CHR_FILE      path to a chr file" << endl;
    cout << "READ_FILE     path to a read file" << endl;
}

int main(int argc, char **argv)
{
    // initialize MPI
    int comm_sz, rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n_reads = 0;            // n's allowed in reads
    int n_chr = 0;              // n's allowed in chrs
    int chr_count;              // number of chr files
    vector<string> chr_files;   // chr file names (all ranks need this)

    if (rank == 0) {
        // consume the first arg
        --argc; ++argv;

        if (argc < 6) {
            print_usage();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        cout << "HERE" << endl;
        cout << "ARGC: " << argc << endl;

        // read in the n-max for reads and chr
        if ((n_reads = strtol(*argv++, NULL, 10)) < 0) {
            print_usage();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        if ((n_chr = strtol(*argv++, NULL, 10)) < 0) {
            print_usage();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        // parse our file args
        vector<string> read_files;
        if (!parse_args(argc-2, argv, chr_files, read_files)) {
            print_usage();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        cout << "Read files: " << read_files.size() << endl;

        // read in our read files
        vector<char> reads;
        fstream fs;
        char buf[BUF_LEN];
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
                fs.getline(buf, BUF_LEN);

                // terminate if not good read
                if (!fs.good()) break;

                // insert the line (WITH '\0') to our reads vector
                stringToUpper(buf, strnlen(buf, BUF_LEN));
                if (isValid(*buf) && nCount(buf, READ_LEN) < n_reads)
                    reads.insert(reads.end(), buf, buf+READ_LEN+1);
            }

            // close the file and go to the next one
            fs.close();
        }

        // print out our reads
        cout << "Reads length: " << reads.size() << endl;
        chr_count = chr_files.size();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&chr_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    cout << "Process #" << rank << ": " << chr_count << endl;

    // go through the chr files
    char filename[BUF_LEN];
    char buf[CHR_LEN*CHR_COUNT];
    MPI_File fh;
    MPI_Status status;
    for (int i = 0; i < chr_count; ++i) {
        // send the filename to all other nodes to open
        if (rank == 0)
            strncpy(filename, chr_files.at(i).c_str(), BUF_LEN);
        MPI_Bcast(&filename, strnlen(filename, BUF_LEN-1)+1, MPI_CHAR, 0, MPI_COMM_WORLD);

        // open the file on all nodes
        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        
        // do this continually in the future
        MPI_File_read_all(fh, buf, CHR_LEN*CHR_COUNT, MPI_CHAR, &status);

        // close the file
        MPI_File_close(&fh);
    }

    MPI_Finalize();
    return 0; 
}
