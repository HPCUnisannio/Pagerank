#include "data.h"

static const Graph TEST = {
    .nodes = 6,
    .edges = 19,
    .filepath = "dataset/data0.dat"
};


static const Graph SMALL = {
    .nodes = 4039,
    .edges = 176468,
    .filepath = "dataset/data1.dat"
};

static const Graph MEDIUM = {
    .nodes = 685230,
    .edges = 7600595,
    .filepath = "dataset/data2.dat"
};

static const Graph BIGGEST = {
    .nodes = 4847571,
    .edges = 68993773,
    .filepath = "dataset/data3.dat"
};

const Graph* get_graph(GraphType type) {
    switch (type) {
        case GRAPH_TEST:
            return &TEST;
        case GRAPH_SMALL:
            return &SMALL;
        case GRAPH_MEDIUM:
            return &MEDIUM;
        case GRAPH_BIGGEST:
            return &BIGGEST;
        default:
            return &TEST; // Default al test se il tipo non è riconosciuto
    }
}