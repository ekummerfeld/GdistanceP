package edu.cmu.tetrad.algcomparison.simulation;

import edu.cmu.tetrad.algcomparison.graph.RandomGraph;
import edu.cmu.tetrad.bayes.BayesIm;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.MlBayesIm;
import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.sem.ParamType;
import edu.cmu.tetrad.sem.Parameter;
import edu.cmu.tetrad.sem.SemPm;
import edu.cmu.tetrad.util.Parameters;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.dist.Discrete;
import edu.cmu.tetrad.util.dist.Indicator;
import org.apache.commons.lang3.RandomUtils;

import java.util.*;

/**
 * A simulation method based on the mixed variable polynomial assumption.
 *
 * @author Bryan Andrews
 */
public class MVPSimulation implements Simulation {
    static final long serialVersionUID = 23L;
    private RandomGraph randomGraph;
    private List<DataSet> dataSets = new ArrayList<>();
    private List<Graph> graphs = new ArrayList<>();
    private DataType dataType;
    private List<Node> shuffledOrder;

    private double interceptLow = 0;
    private double interceptHigh = 1;
    private double continuousInfluence = 0.5;
    private double linearLow = 0;
    private double linearHigh = 1;
    private double quadraticLow = 0;
    private double quadraticHigh = 1;
    private double cubicLow = 0;
    private double cubicHigh = 0.5;
    private double varLow = 0.5;
    private double varHigh = 0.5;

    public MVPSimulation(RandomGraph graph) {
        this.randomGraph = graph;
    }

    @Override
    public void createData(Parameters parameters) {
        setInterceptLow(parameters.getDouble("interceptLow"));
        setInterceptHigh(parameters.getDouble("interceptHigh"));
        setContinuousInfluence(parameters.getDouble("continuousInfluence"));
        setLinearLow(parameters.getDouble("linearLow"));
        setLinearHigh(parameters.getDouble("linearHigh"));
        setQuadraticLow(parameters.getDouble("quadraticLow"));
        setQuadraticHigh(parameters.getDouble("quadraticHigh"));
        setCubicLow(parameters.getDouble("cubicLow"));
        setCubicHigh(parameters.getDouble("cubicHigh"));
        setVarLow(parameters.getDouble("varLow"));
        setVarHigh(parameters.getDouble("varHigh"));

        double percentDiscrete = parameters.getDouble("percentDiscrete");

        boolean discrete = parameters.getString("dataType").equals("discrete");
        boolean continuous = parameters.getString("dataType").equals("continuous");

        if (discrete && percentDiscrete != 100.0) throw new IllegalArgumentException("To simulate discrete data, 'percentDiscrete' must be set to 0.0.");
        else if (continuous && percentDiscrete != 0.0) throw new IllegalArgumentException("To simulate continuoue data, 'percentDiscrete' must be set to 100.0.");

        if (discrete) this.dataType = DataType.Discrete;
        if (continuous) this.dataType = DataType.Continuous;

        this.shuffledOrder = null;

        Graph graph = randomGraph.createGraph(parameters);

        dataSets = new ArrayList<>();
        graphs = new ArrayList<>();

        for (int i = 0; i < parameters.getInt("numRuns"); i++) {
            System.out.println("Simulating dataset #" + (i + 1));

            if (parameters.getBoolean("differentGraphs") && i > 0) graph = randomGraph.createGraph(parameters);

            graphs.add(graph);

            DataSet dataSet = simulate(graph, parameters);
            dataSet.setName("" + (i + 1));
            dataSets.add(dataSet);
        }
    }

    @Override
    public Graph getTrueGraph(int index) {
        return graphs.get(index);
    }

    @Override
    public DataSet getDataSet(int index) {
        return dataSets.get(index);
    }

    @Override
    public String getDescription() {
        return "MVP simulation using " + randomGraph.getDescription();
    }

    @Override
    public List<String> getParameters() {
        List<String> parameters = randomGraph.getParameters();
        parameters.add("minCategories");
        parameters.add("maxCategories");
        parameters.add("percentDiscrete");
        parameters.add("numRuns");
        parameters.add("differentGraphs");
        parameters.add("sampleSize");
        parameters.add("interceptLow");
        parameters.add("interceptHigh");
        parameters.add("continuousInfluence");
        parameters.add("linearLow");
        parameters.add("linearHigh");
        parameters.add("quadraticLow");
        parameters.add("quadraticHigh");
        parameters.add("cubicLow");
        parameters.add("cubicHigh");
        parameters.add("varLow");
        parameters.add("varHigh");
        return parameters;
    }

    @Override
    public int getNumDataSets() {
        return dataSets.size();
    }

    @Override
    public DataType getDataType() {
        return dataType;
    }

    private DataSet simulate(Graph G, Parameters parameters) {
        HashMap<String, Integer> nd = new HashMap<>();

        List<Node> nodes = G.getNodes();

        Collections.shuffle(nodes);

        if (this.shuffledOrder == null) {
            List<Node> shuffledNodes = new ArrayList<>(nodes);
            Collections.shuffle(shuffledNodes);
            this.shuffledOrder = shuffledNodes;
        }

        for (int i = 0; i < nodes.size(); i++) {
            if (i < nodes.size() * parameters.getDouble("percentDiscrete") * 0.01) {
                final int minNumCategories = parameters.getInt("minCategories");
                final int maxNumCategories = parameters.getInt("maxCategories");
                final int value = pickNumCategories(minNumCategories, maxNumCategories);
                nd.put(shuffledOrder.get(i).getName(), value);
            } else {
                nd.put(shuffledOrder.get(i).getName(), 0);
            }
        }

        G = makeMixedGraph(G, nd);
        nodes = G.getNodes();

        DataSet mixedData = new BoxDataSet(new MixedDataBox(nodes, parameters.getInt("sampleSize")), nodes);

        List<Node> tierOrdering = G.getCausalOrdering();
        int[] tiers = new int[tierOrdering.size()];
        for (int t = 0; t < tierOrdering.size(); t++) {
            tiers[t] = nodes.indexOf(tierOrdering.get(t));
        }

        for (int mixedIndex : tiers) {

            HashMap<Integer, Double> noiseVar = new HashMap<>();

            if (nodes.get(mixedIndex) instanceof DiscreteVariable) {

                DiscreteVariable child = (DiscreteVariable) nodes.get(mixedIndex);
                ArrayList<DiscreteVariable> discreteParents = new ArrayList<>();
                ArrayList<ContinuousVariable> continuousParents = new ArrayList<>();
                for (Node node : G.getParents(child)) {
                    if (node instanceof DiscreteVariable) discreteParents.add((DiscreteVariable) node);
                    else continuousParents.add((ContinuousVariable) node);
                }

                HashMap<String, double[]> intercept = new HashMap<>();
                HashMap<String, double[]> linear = new HashMap<>();
                HashMap<String, double[]> quadratic = new HashMap<>();
                HashMap<String, double[]> cubic = new HashMap<>();

                double[][] probs = new double[parameters.getInt("sampleSize")][child.getNumCategories()];
                double[][] intercepts = new double[parameters.getInt("sampleSize")][child.getNumCategories()];
                double min = 0;
                double max = 0;

                for (int i = 0; i < parameters.getInt("sampleSize"); i++) {

                    double[] parents = new double[continuousParents.size()];

                    for (int category = 0; category < child.getNumCategories(); category++) {
                        String key = ((Integer) category).toString().concat(",");
                        for (int j = 1; j <= discreteParents.size(); j++) {
                            key = key.concat(((Integer) mixedData.getInt(i, mixedData.getColumn(discreteParents.get(j - 1)))).toString()).concat(",");
                        }
                        for (int j = 1; j <= continuousParents.size(); j++)
                            parents[j - 1] = mixedData.getDouble(i, mixedData.getColumn(continuousParents.get(j - 1)));

                        if (!intercept.containsKey(key)) {
                            double[] interceptCoefficients = new double[child.getNumCategories()];
                            double norm = 0;
                            for (int childCategory = 0; childCategory < child.getNumCategories(); childCategory++) {
                                interceptCoefficients[childCategory] = RandomUtil.getInstance().nextGamma(1,1);
                                norm += interceptCoefficients[childCategory];
                            }
                            for (int childCategory = 0; childCategory < child.getNumCategories(); childCategory++) {
                                interceptCoefficients[childCategory] /= norm;
                            }
                            for (int childCategory = 0; childCategory < child.getNumCategories(); childCategory++) {
                                String temp = ((Integer) childCategory).toString().concat(",");
                                for (int j = 1; j <= discreteParents.size(); j++) {
                                    temp = temp.concat(((Integer) mixedData.getInt(i, mixedData.getColumn(discreteParents.get(j - 1)))).toString()).concat(",");
                                }
                                double[] interceptCoefficient = new double[1];
                                interceptCoefficient[0] = interceptCoefficients[childCategory];
                                intercept.put(temp, interceptCoefficient);
                            }
                        }

                        if (!linear.containsKey(key) && !continuousParents.isEmpty()) {
                            double[] linearCoefficients = new double[parents.length];
                            for (int j = 0; j < parents.length; j++)
                                linearCoefficients[j] = randSign() * RandomUtil.getInstance().nextUniform(linearLow, linearHigh);
                            linear.put(key, linearCoefficients);
                        }

                        if (!quadratic.containsKey(key) && !continuousParents.isEmpty()) {
                            double[] quadraticCoefficients = new double[parents.length];
                            for (int j = 0; j < parents.length; j++)
                                quadraticCoefficients[j] = randSign() * RandomUtil.getInstance().nextUniform(quadraticLow, quadraticHigh);
                            quadratic.put(key, quadraticCoefficients);
                        }

                        if (!cubic.containsKey(key) && !continuousParents.isEmpty()) {
                            double[] cubicCoefficients = new double[parents.length];
                            for (int j = 0; j < parents.length; j++)
                                cubicCoefficients[j] = linear.get(key)[j]/Math.abs(linear.get(key)[j]) * RandomUtil.getInstance().nextUniform(cubicLow, cubicHigh);
                            cubic.put(key, cubicCoefficients);
                        }

                        intercepts[i][category] = intercept.get(key)[0];
                        probs[i][category] = 0;
                        for (int x = 0; x < parents.length; x++) {
                            probs[i][category] += linear.get(key)[x] * parents[x];
                            probs[i][category] += quadratic.get(key)[x] * Math.pow(parents[x], 2);
                            probs[i][category] += cubic.get(key)[x] * Math.pow(parents[x], 3);
                        }
                        min = Math.min(min, probs[i][category]);
                        max = Math.max(max, probs[i][category]);
                    }
                }

                for (int i = 0; i < parameters.getInt("sampleSize"); i++) {
                    double norm = 0;
                    if (max == 0) { max = 1; }
                    for (int category = 0; category < child.getNumCategories(); category++)
                        norm += continuousInfluence * (probs[i][category] - min)/(max - min) + intercepts[i][category];
                    double prob = RandomUtil.getInstance().nextUniform(0, norm);
                    for (int category = 0; category < child.getNumCategories(); category++) {
                        prob -= continuousInfluence * (probs[i][category] - min)/(max - min) + intercepts[i][category];
                        if (prob <= 0) {
                            mixedData.setInt(i, mixedIndex, category);
                            break;
                        }
                    }
                }

            } else {

                noiseVar.put(mixedIndex, RandomUtil.getInstance().nextUniform(varLow, varHigh));

                ContinuousVariable child = (ContinuousVariable) nodes.get(mixedIndex);
                ArrayList<DiscreteVariable> discreteParents = new ArrayList<>();
                ArrayList<ContinuousVariable> continuousParents = new ArrayList<>();
                for (Node node : G.getParents(child)) {
                    if (node instanceof DiscreteVariable) discreteParents.add((DiscreteVariable) node);
                    else continuousParents.add((ContinuousVariable) node);
                }

                HashMap<String, double[]> intercept = new HashMap<>();
                HashMap<String, double[]> linear = new HashMap<>();
                HashMap<String, double[]> quadratic = new HashMap<>();
                HashMap<String, double[]> cubic = new HashMap<>();

                double mean = 0;
                double var = 0;

                boolean resample;
                boolean orders = true;

                do {

                    resample = false;

                    for (int i = 0; i < parameters.getInt("sampleSize"); i++) {

                        double[] parents = new double[continuousParents.size()];
                        double value = 0;
                        String key = "";
                        for (int j = 1; j <= discreteParents.size(); j++) {
                            key = key.concat(((Integer) mixedData.getInt(i, mixedData.getColumn(discreteParents.get(j - 1)))).toString()).concat(",");
                        }
                        for (int j = 1; j <= continuousParents.size(); j++)
                            parents[j - 1] = mixedData.getDouble(i, mixedData.getColumn(continuousParents.get(j - 1)));

                        if (!intercept.containsKey(key)) {
                            double[] interceptCoefficients = new double[1];
                            interceptCoefficients[0] = randSign() * RandomUtil.getInstance().nextUniform(interceptLow, interceptHigh);
                            intercept.put(key, interceptCoefficients);
                        }

                        if (!linear.containsKey(key) && !continuousParents.isEmpty()) {
                            double[] linearCoefficients = new double[parents.length];
                            for (int j = 0; j < parents.length; j++)
                                linearCoefficients[j] = randSign() * RandomUtil.getInstance().nextUniform(linearLow, linearHigh);
                            linear.put(key, linearCoefficients);
                        }

                        if (!quadratic.containsKey(key) && !continuousParents.isEmpty()) {
                            double[] quadraticCoefficients = new double[parents.length];
                            for (int j = 0; j < parents.length; j++)
                                quadraticCoefficients[j] = randSign() * RandomUtil.getInstance().nextUniform(quadraticLow, quadraticHigh);
                            quadratic.put(key, quadraticCoefficients);
                        }

                        if (!cubic.containsKey(key) && !continuousParents.isEmpty()) {
                            double[] cubicCoefficients = new double[parents.length];
                            for (int j = 0; j < parents.length; j++)
                                cubicCoefficients[j] = randSign() * RandomUtil.getInstance().nextUniform(cubicLow, cubicHigh);
                            cubic.put(key, cubicCoefficients);
                        }

                        value += intercept.get(key)[0];
                        if (!continuousParents.isEmpty()) {
                            for (int x = 0; x < parents.length; x++) {
                                if (orders) {
                                    value += linear.get(key)[x] * parents[x];
                                    value += quadratic.get(key)[x] * Math.pow(parents[x], 2);
                                    value += cubic.get(key)[x] * Math.pow(parents[x], 3);
                                } else {
                                    double sign = parents[x] / Math.abs(parents[x]);
                                    value += (1/quadratic.get(key)[x]) * sign * Math.pow(Math.abs(parents[x]), (1 / 2));
                                    value += (1/cubic.get(key)[x]) * sign * Math.pow(Math.abs(parents[x]), (1 / 3));
                                }
                            }
                            mean += value;
                            var += Math.pow(value, 2);
                        }
                        mixedData.setDouble(i, mixedIndex, value);
                    }
                    if (!continuousParents.isEmpty()) {
                        mean /= mixedData.getNumRows();
                        var /= mixedData.getNumRows();
                        var -= Math.pow(mean, 2);
                        var = Math.sqrt(var);
                    } else {
                        var = 1;
                    }

                    for (int i = 0; i < mixedData.getNumRows(); i++) {
                        double value = (mixedData.getDouble(i, mixedIndex) - mean) / var;
                        mixedData.setDouble(i, mixedIndex, value + RandomUtil.getInstance().nextNormal(0, noiseVar.get(mixedIndex)));
                        if (Math.abs(value) > 10 && orders) {
                            orders = false;
                            resample = true;
                            mean = 0;
                            var = 0;
                            break;
                        }
                    }
                } while (resample);
            }
        }
        return mixedData;
    }


    public void setInterceptLow(double interceptLow) {
        this.interceptLow = interceptLow;
    }

    public void setInterceptHigh(double interceptHigh) {
        this.interceptHigh = interceptHigh;
    }

    public void setContinuousInfluence(double continuousInfluence) {this.continuousInfluence = continuousInfluence; }

    public void setLinearLow(double linearLow){
        this.linearLow = linearLow;
    }

    public void setLinearHigh(double linearHigh) {
        this.linearHigh = linearHigh;
    }

    public void setQuadraticLow(double quadraticLow) {
        this.quadraticLow = quadraticLow;
    }

    public void setQuadraticHigh(double quadraticHigh) {
        this.quadraticHigh = quadraticHigh;
    }

    public void setCubicLow(double cubicLow) {
        this.cubicLow = cubicLow;
    }

    public void setCubicHigh(double  cubicHigh) {
        this.cubicHigh = cubicHigh;
    }

    public void setVarLow(double varLow) {
        this.varLow = varLow;
    }

    public void setVarHigh(double varHigh) {
        this.varHigh = varHigh;
    }

    private int randSign() { return RandomUtil.getInstance().nextInt(2)*2-1; }

    private static Graph makeMixedGraph(Graph g, Map<String, Integer> m) {
        List<Node> nodes = g.getNodes();
        for (int i = 0; i < nodes.size(); i++) {
            Node n = nodes.get(i);
            int nL = m.get(n.getName());
            if (nL > 0) {
                Node nNew = new DiscreteVariable(n.getName(), nL);
                nodes.set(i, nNew);
            } else {
                Node nNew = new ContinuousVariable(n.getName());
                nodes.set(i, nNew);
            }
        }

        Graph outG = new EdgeListGraph(nodes);

        for (Edge e : g.getEdges()) {
            Node n1 = e.getNode1();
            Node n2 = e.getNode2();
            Edge eNew = new Edge(outG.getNode(n1.getName()), outG.getNode(n2.getName()), e.getEndpoint1(), e.getEndpoint2());
            outG.addEdge(eNew);
        }

        return outG;
    }

    private int pickNumCategories(int min, int max) {
        return RandomUtils.nextInt(min, max + 1);
    }
}
