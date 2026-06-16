
#ifndef PAGERANK_GRAPH_H
#define PAGERANK_GRAPH_H

typedef struct {
    int nodes;
    int edges;
    const char *filepath;
} Graph;

typedef enum
{
    GRAPH_TEST,
    GRAPH_SMALL,
    GRAPH_MEDIUM,
    GRAPH_BIGGEST
} GraphType;

const Graph* get_graph(GraphType type);
#endif
