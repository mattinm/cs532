#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <unordered_map>
#include <vector>

#define READ_LEN    40
#define CHR_LEN     51
#define CHR_COUNT   100
#define CHR_BUF     (CHR_LEN * CHR_COUNT)
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

/** Checks to see if the string is valid.
 * @param s the string to check
 * @param n the length of the string
 */
bool isValid(char *s, int n)
{
    // if any characters don't match, return false
    while (n--) {
        if (*s != 'N' && *s != 'A' && *s != 'C' && *s != 'G' && *s != 'T')
            return false;
        ++s;
    }

    // all characters matched
    return true;
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

int nLocation(char *s, int n)
{
    for (int i = 0; i < n; ++i)
        if (s[i] == 'N') return i;

    return -1;
}

/** Add a string to the readmap, or increment the value at the location.
 * Converts Ns to all 4 possible values.
 */
void addToReadmap(char *s, int len, unordered_map<string, int> &readmap)
{
    // if we have an N, we need to replace it with all possible values,
    // the recursively call addToReadmap
    int nloc = nLocation(s, len);
    if (nloc >= 0) {
        char *s2 = new char[len];
        memcpy(s2, s, len);
        
        s2[nloc] = 'A';
        addToReadmap(s2, len, readmap);

        s2[nloc] = 'C';
        addToReadmap(s2, len, readmap);

        s2[nloc] = 'G';
        addToReadmap(s2, len, readmap);

        s2[nloc] = 'T';
        addToReadmap(s2, len, readmap);

        delete[] s2;
        return;
    }

    // since we don't have Ns, just add to the readmap
    string str(s, len);
    auto it = readmap.find(str);
    if (it != readmap.end())
        it->second += 1;
    else
        readmap.insert({{str, 1}});
}

void addToChrMap(char *s, int len, int index, const unordered_map<string, int> &readmap, vector<int> &indexes, vector<int> &counts)
{
    int nloc = nLocation(s, len);
    if (nloc >= 0) {
        char *s2 = new char[len];
        memcpy(s2, s, len);
        
        s2[nloc] = 'A';
        addToChrMap(s2, len, index, readmap, indexes, counts);

        s2[nloc] = 'C';
        addToChrMap(s2, len, index, readmap, indexes, counts);

        s2[nloc] = 'G';
        addToChrMap(s2, len, index, readmap, indexes, counts);

        s2[nloc] = 'T';
        addToChrMap(s2, len, index, readmap, indexes, counts);

        delete[] s2;
        return;
    }

    string str(s, len);
    auto it = readmap.find(str);
    if (it != readmap.end()) {
        indexes.push_back(index);
        counts.push_back(it->second);
    }
}

/** Prints out the usage of the program. */
void print_usage()
{
    cout << "Usage: ./genome OUTPUT_FILE N_READS N_CHR --chr CHR_FILE [CHR_FILE ...] --read READ_FILE [READ_FILE ...]" << endl;
    cout << "============================================================================================" << endl;
    cout << "OUTPUT_FILE   file to output to" << endl;
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
    int readslen;
    string outfilename;
    vector<string> chr_files;   // chr file names (all ranks need this)
    string reads;
    fstream fs;

    if (rank == 0) {
        // consume the first arg
        --argc; ++argv;

        if (argc < 7) {
            print_usage();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        outfilename = *argv++;

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
        if (!parse_args(argc-3, argv, chr_files, read_files)) {
            print_usage();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        cout << endl << "READ Files" << endl;
        cout << "==========" << endl;
        cout << "Count: " << read_files.size() << endl;

        // read in our read files
        char buf[BUF_LEN];
        for (auto it = read_files.begin(); it != read_files.end(); ++it) {
            // open the file
            string filename = *it;
            cout << "Opening file: " << filename << endl;
            fs.open(filename, fstream::in);
            if (!fs.is_open()) {
                cout << "\tFailed to open file: " << filename;
                continue;
            }

            // read in all our required lines
            while (1) {
                fs.getline(buf, BUF_LEN);

                // terminate if not good read
                if (!fs.good()) break;

                // continue if this isn't the correct length
                if (strlen(buf) != READ_LEN) continue;

                stringToUpper(buf, READ_LEN);
                if (isValid(buf, READ_LEN) && nCount(buf, READ_LEN) <= n_reads) {
                    reads.append(buf, READ_LEN);
                    reads.push_back('\0');
                }
            }

            // close the file and go to the next one
            fs.close();
        }

        // print out our reads
        cout << "Reads length: " << reads.size() << endl;
        readslen = reads.size();
        chr_count = chr_files.size();
    }

    MPI_Bcast(&chr_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_chr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&readslen, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // split out the reads to the different nodes
    int slices = comm_sz;
    int slicesz = readslen / (READ_LEN+1);
    int slicelen = slicesz / slices;
    int firstlen = slicelen + slicesz - slicelen * slices;

    // assign our initial slize / displacement for 0
    int *slice_szs = new int[comm_sz];
    int *displacements = new int[comm_sz];
    *slice_szs = firstlen * (READ_LEN + 1);
    *displacements = 0;

    // assign each slice/displacement
    for (int i = 1; i < comm_sz; ++i) {
        slice_szs[i] = slicelen * (READ_LEN + 1);
        displacements[i] = displacements[i-1] + slice_szs[i-1];
    }

    if (rank == 0) slicelen = firstlen;

    // scatter out our slices
    char *slice = new char[slice_szs[rank]];
    MPI_Scatterv(
            reads.data(),
            slice_szs,
            displacements,
            MPI_CHAR,
            slice,
            slice_szs[rank],
            MPI_CHAR,
            0,
            MPI_COMM_WORLD
    );

    reads.clear();

    // fill an unordered map with the strings
    unordered_map<string, int> readmap;
    for (int i = 0; i < slice_szs[rank]; i += READ_LEN+1)
        addToReadmap(&slice[i], READ_LEN, readmap);

    if (rank == 0) {
        cout << endl << "CHR Files" << endl;
        cout << "==========" << endl;
        cout << "CHR Count: " << chr_count << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // go through the chr files
    char filename[BUF_LEN];
    char buf[CHR_BUF + READ_LEN];
    int readcount;
    int icount;
    bool lastvalid;
    MPI_File fh;
    MPI_Status status;
    MPI_Offset filesize;
    MPI_Offset bytesread;
    for (int i = 0; i < chr_count; ++i) {
        // send the filename to all other nodes to open
        if (rank == 0) {
            strncpy(filename, chr_files.at(i).c_str(), BUF_LEN);
            filename[BUF_LEN-1] = '\0';
            cout << "File #" << i << ": " << filename << endl;
        }
        MPI_Bcast(filename, BUF_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

        // some debug info
        cout << "\tProcess #" << rank << ": " << filename << endl;

        // open the file on all nodes
        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

        // determine the file length so we know how many times to read
        MPI_File_get_size(fh, &filesize);
        bytesread = 0;
        readcount = CHR_BUF + READ_LEN;
        if ((int)(filesize) < readcount)
            readcount = (int)(filesize);
    
        // read in the first buf amount
        lastvalid = false;
        MPI_File_read_all(fh, buf, readcount, MPI_CHAR, &status);
        bytesread += readcount;

        vector<int> indexes;
        vector<int> counts;

        if (rank == 0) {
            fs.open(outfilename, fstream::out);
            if (!fs.is_open()) {
                cout << "FAILED TO OPEN " << outfilename << endl;
            }
        }

        // keep reading in until end of file
        do {
            // process each READ_LEN amount from our buffer
            stringToUpper(buf, readcount);
            icount = readcount - READ_LEN;
            for (int j = 0; j < icount; ++j) {
                // if our last was valid, we only need to check the next
                // character; otherwise, we need to check all of them
                if (lastvalid) {
                    if (isValid(&buf[j+READ_LEN-1], 1) && nCount(&buf[j], READ_LEN) <= n_chr) {
                        lastvalid = true;
                    } else {
                        lastvalid = false;
                    }
                } else if (isValid(&buf[j], READ_LEN) && nCount(&buf[j], READ_LEN) <= n_chr) {
                    lastvalid = true;
                } else {
                    lastvalid = false;
                }

                // if we have a valid string, use the recursive addToChrMap function
                // to check Ns and update our indexes and counts
                if (lastvalid)
                    addToChrMap(&buf[j], READ_LEN, bytesread + j, readmap, indexes, counts);
            }

            if (bytesread >= filesize) break;

            if ((int)(filesize - bytesread) < readcount)
                readcount = (int)(filesize - bytesread);

            memmove(buf, &buf[CHR_BUF], READ_LEN);
            MPI_File_read_all(fh, &buf[READ_LEN], readcount, MPI_CHAR, &status);
            bytesread += readcount;
        } while (1);

        // determine the max size of our array required for sending/receiving
        int size = indexes.size();
        int max = 0;
        MPI_Reduce(&size, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            // send out the data 
            MPI_Send(indexes.data(), indexes.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(counts.data(), counts.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
        } else {
            int received = 1;
            int *rindexes = new int[max];
            int *rcounts = new int[max];
            int actualcount = 0;
            map<int, int> curmap; 

            cout << endl << "Received " << indexes.size() << " results from process #0" << endl;
            for (auto it = indexes.begin(), it2 = counts.begin(); it != indexes.end(); ++it, ++it2)
                curmap.insert(pair<int, int>(*it, *it2));

            while (received++ < comm_sz) {
                // recieve the index array from any source, but the counts array
                // from the same source
                MPI_Recv(rindexes, max, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(rcounts, max, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);

                MPI_Get_count(&status, MPI_INT, &actualcount);
                cout << "Recieved " << actualcount << " results from process #" << status.MPI_SOURCE << endl;

                // add the data into our map
                for (int j = 0; j < actualcount; ++j) {
                    auto it = curmap.find(rindexes[j]);
                    if (it != curmap.end())
                        it->second += rcounts[j];
                    else
                        curmap.insert(pair<int, int>(rindexes[j], rcounts[j]));
                }
            }

            // print out our findings
            for (auto it = curmap.begin(); it != curmap.end(); ++it) {
                if (fs.is_open())
                    fs << chr_files.at(i).c_str() << ", " << it->first << ", " << it->second << endl;
                else
                    cerr << chr_files.at(i).c_str() << ", " << it->first << ", " << it->second << endl;
            }

            // free our memory
            delete[] rindexes;
            delete[] rcounts;
            curmap.clear();
        }
        
        // close the file
        MPI_File_close(&fh);
    }

    // free memory
    delete[] slice_szs;
    delete[] displacements;
    delete[] slice;

    // finalize and return
    MPI_Finalize();
    return 0; 
}
