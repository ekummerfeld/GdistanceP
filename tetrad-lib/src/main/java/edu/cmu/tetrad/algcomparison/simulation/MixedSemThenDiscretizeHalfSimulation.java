package edu.cmu.tetrad.algcomparison.simulation;

import edu.cmu.tetrad.algcomparison.DataType;
import edu.cmu.tetrad.algcomparison.Parameters;
import edu.cmu.tetrad.algcomparison.Simulation;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.Discretizer;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.sem.SemIm;
import edu.cmu.tetrad.sem.SemPm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by jdramsey on 6/4/16.
 */
public class MixedSemThenDiscretizeHalfSimulation implements Simulation {
    private List<DataSet> dataSets;
    private Graph graph;

    public MixedSemThenDiscretizeHalfSimulation(Parameters parameters) {
        Graph graph = GraphUtils.scaleFreeGraph(
                parameters.getInt("numMeasures"),
                parameters.getInt("numLatents"),
                parameters.getDouble("scaleFreeAlpha"),
                parameters.getDouble("scaleFreeBeta"),
                parameters.getDouble("scaleFreeDeltaIn"),
                parameters.getInt("scaleFreeDeltaOut")
        );

        dataSets = new ArrayList<>();

        for (int i = 0; i < parameters.getInt("numRuns"); i++) {
            dataSets.add(simulate(graph, parameters));
        }
    }

    public Graph getTrueGraph() {
        return graph;
    }

    public String getDescription() {
        return "Simulation SEM data then discretizing some variables";
    }

    @Override
    public int getNumDataSets() {
        return dataSets.size();
    }

    @Override
    public DataSet getDataSet(int index) {
        return dataSets.get(index);
    }

    @Override
    public DataType getDataType() {
        return DataType.Mixed;
    }

    private DataSet simulate(Graph graph, Parameters parameters) {
        SemPm pm = new SemPm(graph);
        SemIm im = new SemIm(pm);
        DataSet continuousData = im.simulateData(parameters.getInt("sampleSize"), false);

        List<Node> shuffledNodes = new ArrayList<>(continuousData.getVariables());
        Collections.shuffle(shuffledNodes);

        Discretizer discretizer = new Discretizer(continuousData);

        for (int i = 0; i < shuffledNodes.size() * parameters.getDouble("percentDiscreteForMixedSimulation") * 0.01; i++) {
            discretizer.equalIntervals(shuffledNodes.get(i), parameters.getInt("numCategories"));
        }

        return discretizer.discretize();
    }
}
