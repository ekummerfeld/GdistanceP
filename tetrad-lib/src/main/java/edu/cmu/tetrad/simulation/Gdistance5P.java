package edu.cmu.tetrad.simulation;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.util.ForkJoinPoolInstance;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

/**
 * Created by Erich on 7/3/2016.
 *
 * This class is used to compare the distance between two graphs learned from fmri data
 * the distance is calculated as the mean of the distance of the edges between the graphs
 * the distance between two edges is calculated as the distance between their endpoints
 * the distance between edges calculated this way is a true distance
 * the distance between two graphs is not a true distance because it is not symmetric
 * the 5P version allows for non-cubic voxels, and parallelizes the most expensive loop
 */
public class Gdistance5P {

    private DataSet locationMap;
    private double xDist;
    private double yDist;
    private double zDist;

    private int chunksize = 2;

    //With the parallel version, it is better to make a constructor for central data like locationMap
    public Gdistance5P(DataSet locationMap, double xDist, double yDist, double zDist){
        this.locationMap = locationMap;
        this.xDist=xDist;
        this.yDist=yDist;
        this.zDist=zDist;
    }

    public List<Double> distances(Graph graph1, Graph graph2) {
        //compared to GdistanceVic, Gdistance5 needs to calculate distances for non-cubic voxels.
        //dimensions along each dimension should be given as input: xdist, ydist, zdist
        //this impliments a less brute force approach, where edge comparisons are restricted
        //to edges that are in the "vicinity" of the original edge

        List<Double> leastList = new ArrayList<>();
        //System.out.println(locationMap);
        // Make *SURE* that the graph nodes are the same as the location nodes
        System.out.println("Synchronizing variables between graph1, graph2, and the locationMap");
        graph1 = GraphUtils.replaceNodes(graph1,locationMap.getVariables());
        graph2 = GraphUtils.replaceNodes(graph2,locationMap.getVariables());

        //constructing vicinity is costy, so do it just once, OUTSIDE the first for loop
        //Using EK's Vicinity5 for testing purposes
        System.out.println("Constructing vicinity object");
        long timevic1 = System.nanoTime();
        ArrayList<Edge> graph2edges = new ArrayList<>(graph2.getEdges());
        Vicinity5 vicinity = new Vicinity5(graph2edges,locationMap,0,100,0,100,0,100,xDist,yDist,zDist);
        long timevic2 = System.nanoTime();
        System.out.println("Done constructing vicinity object. Construction Time : " + (timevic2 - timevic1)/1000000000 + "s" );

        //This first for loop should be parallelized in the future.
        //let the for loop do its thing, and create a new thread for each task inside of it.
        int edgetracker=1;
        ForkJoinPool pool = ForkJoinPoolInstance.getInstance().getPool();
        for (Edge edge1 : graph1.getEdges()) {
            System.out.println("edge#"+edgetracker); edgetracker++;
            //the above should probably be left, everything below should be turned into a recursive task
            // for each choice we will create a task that will run on a separate thread
            pool.invoke(new FindLeastDistanceTask(leastList, vicinity, edge1));

        }
        return leastList;
    }

    //////+++++******* Method used in multithread task
    class FindLeastDistanceTask extends RecursiveTask<Boolean> {

        List<Double> leastList;
        Vicinity5 vicinity;
        Edge edge1;


        public FindLeastDistanceTask(final List<Double> leastList, final Vicinity5 vicinity, final Edge edge1) {
            this.leastList = leastList;
            this.vicinity = vicinity;
            this.edge1=edge1;

        }

        @Override
        protected Boolean compute() {

            //the variable "count" is used to initialize leastDistance to the first thisDistance
            int count = 1;
            double thisDistance;
            double leastDistance = -1.0;
            //the next for loop gets restricted to edges in the vicinity of edge1
            List<Edge> vicEdges = vicinity.getVicinity(edge1,chunksize);
            //System.out.println(vicEdges);
            for (Edge edge2 : vicEdges) {
                thisDistance = edgesDistance(edge1, edge2, locationMap,xDist,yDist,zDist);
                //remember only the shortest distance seen
                if (count ==1) {
                    leastDistance = thisDistance;
                } else {
                    if (thisDistance < leastDistance) {
                        leastDistance = thisDistance;
                    }
                }
                count++;
            }
            //add it to a list of the leastDistances
            leastList.add(leastDistance);

            return true;
        }
    }

    //////======***PRIVATE METHODS BELOW *****=====/////

    private static double nodesDistance(Node node1, Node node2, DataSet locationMap, double x, double y, double z) {
        //calculate distance between two nodes based on their locations
        //simple starter is simply the taxicab distance:
        //calc differences in X, Y, and Z axis, then sum them together.
        int column1 = locationMap.getColumn(node1);
        int column2 = locationMap.getColumn(node2);

        //System.out.println(column1);

        double value11 = locationMap.getDouble(0,column1);
        double value12 = locationMap.getDouble(1,column1);
        double value13 = locationMap.getDouble(2,column1);

        double value21 = locationMap.getDouble(0,column2);
        double value22 = locationMap.getDouble(1,column2);
        double value23 = locationMap.getDouble(2,column2);

        //taxicab distance
        //double taxicab = Math.abs(value11 - value21) + Math.abs(value12 - value22) + Math.abs(value13 - value23);
        //euclidian distance instead of taxicab
        double euclid = Math.sqrt((value11 - value21)*x*(value11 - value21)*x+(value12 - value22)*y*
                (value12 - value22)*y+(value13 - value23)*z*(value13 - value23)*z );

        return euclid;
    }

    private static double edgesDistance(Edge edge1, Edge edge2, DataSet locationMap, double xD, double yD, double zD) {
        //calculate distance between two edges based on distances of their endpoints
        //if both edges are directed, then:
        //compare edge1 head to edge2 head, tail to tail.
        //sum head distance and tail ditance
        if (edge1.isDirected() && edge2.isDirected()) {
            //find head and tail of edge1
            Node edge1h = Edges.getDirectedEdgeHead(edge1);
            Node edge1t = Edges.getDirectedEdgeTail(edge1);
            //find head and tail of edge2
            Node edge2h = Edges.getDirectedEdgeHead(edge2);
            Node edge2t = Edges.getDirectedEdgeTail(edge2);
            //compare tail to tail
            double tDistance = nodesDistance(edge1t, edge2t, locationMap,xD,yD,zD);
            double hDistance = nodesDistance(edge1h, edge2h, locationMap,xD,yD,zD);
            return tDistance + hDistance;
        }
        else {
            //otherwise if either edge is not directed:
            //for each of edge1's two endpoints, calc distance to both edge2 endpoints
            //store the shorter distances, and sum them.
            Node node11 = edge1.getNode1();
            Node node12 = edge1.getNode2();
            Node node21 = edge2.getNode1();
            Node node22 = edge2.getNode2();

            //first compare node1 to node1 and node2 to node2
            double dist11 = nodesDistance(node11, node21, locationMap,xD,yD,zD);
            double dist22 = nodesDistance(node12, node22, locationMap,xD,yD,zD);

            //then compare node1 to node2 and node2 to node1
            double dist12 = nodesDistance(node11, node22, locationMap,xD,yD,zD);
            double dist21 = nodesDistance(node12, node21, locationMap,xD,yD,zD);

            //then return the minimum of the two ways of pairing nodes from each edge
            return Math.min(dist11 + dist22, dist12 + dist21);
        }

    }
}
