#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"

// Temporary constants
#define DATASET_FILE "data/shakes.txt"

struct app_flags {

    char* dataset_filepath;
};

void app_flags_init(struct app_flags* flags) {
    flags->dataset_filepath = NULL;
}

void app_flags_print(const struct app_flags* flags) {
    printf("Command-line flags:\n");
    printf("\tdataset: %s\n", flags->dataset_filepath);
}

// returns zero on success and attempts to populate
// the app_flags with any command-line flags parsed.
int parse_args(struct app_flags* flags, int argc, char* argv[]) {
    app_flags_init(flags);
    for (int i=1; i<argc; ++i) {
        if (strcmp(argv[i], "--dataset") == 0) {
            // make sure we got one more parameter to read
            if (i+1 <argc) {
                flags->dataset_filepath = argv[i+1];
                i++;
            } else {
                printf("ERROR: no dataset file passed on on the command line.\n");
                return 1;
            }
        }
    }

    return 0;
}

void print_cli_help(char* app_name) {
    printf("\n\nUsage: %s [--dataset <text filepath>]\n\n", app_name);
}

int main(int argc, char* argv[]) {
    // parse the command line flags
    struct app_flags flags;
    int parsed_success = parse_args(&flags, argc, argv);
    if (parsed_success != 0) {
        print_cli_help(argv[0]);
        return 1;
    }

    // print out the detected flags
    app_flags_print(&flags);
}