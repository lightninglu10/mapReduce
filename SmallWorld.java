/*
 *
 * CS61C Spring 2013 Project 2: Small World
 *
 * Partner 1 Name: Ronald Kwan
 * Partner 1 Login: cs61c-ju
 *
 * Partner 2 Name: Patrick Lu
 * Partner 2 Login: cs61c-pi
 *
 * REMINDERS: 
 *
 * 1) YOU MUST COMPLETE THIS PROJECT WITH A PARTNER.
 * 
 * 2) DO NOT SHARE CODE WITH ANYONE EXCEPT YOUR PARTNER.
 * EVEN FOR DEBUGGING. THIS MEANS YOU.
 *
 */

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.Math;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;


public class SmallWorld {
    // Maximum depth for any breadth-first search
    public static final int MAX_ITERATIONS = 20;

    public static class Node implements Writable {
        private long id = -1L;
        private boolean start = false;
        private HashMap<Long, Long> distances;
        private ArrayList<Long> neighbors;

	public Node() {
            neighbors = new ArrayList<Long>();
            distances = new HashMap<Long, Long>();
        }

        public Node(long id) {
	    this.id = id;
            neighbors = new ArrayList<Long>();
            distances = new HashMap<Long, Long>();
        }

        public void setID(long id) {
            this.id = id;
        }

        public long getID() {
            return id;
        }

        public void setStart() {
            start = true;
        }

        public boolean isStart() {
            return start;
        }

        public void setDistance(long root, long dist) {
            distances.put(root, dist);
        }

        public long getDistance(long root) {
	    if (!distances.containsKey(root)) {
		return Long.MAX_VALUE;
	    }
            return distances.get(root);
        }

	public HashMap<Long, Long> getDistances() {
	    return distances;
	}

        public void addNeighbor(long neighbor) {
	    if (!neighbors.contains(neighbor)) {
		neighbors.add(neighbor);
	    }
        }

        public ArrayList<Long> getNeighbors() {
            return neighbors;
        }

        // Serializes object - needed for Writable
        public void write(DataOutput out) throws IOException {
            out.writeLong(id);

	    out.writeBoolean(start);

	    ArrayList<Long> roots = new ArrayList<Long>(distances.keySet());
	    int distlen = roots.size();
	    out.writeInt(distlen);
	    long root, dist;
	    for (int i = 0; i < distlen; i++) {
		root = roots.get(i);
		dist = distances.get(root);
		out.writeLong(root);
		out.writeLong(dist);
	    }

            int neighlen = neighbors.size();
            out.writeInt(neighlen);
            for (int j = 0; j < neighlen; j++){
                out.writeLong(neighbors.get(j));
            }
        }

        // Deserializes object - needed for Writable
        public void readFields(DataInput in) throws IOException {
            id = in.readLong();

	    start = in.readBoolean();

	    int distlen = in.readInt();
	    distances = new HashMap<Long, Long>();
	    long root, dist;
	    for (int i = 0; i < distlen; i++) {
		root = in.readLong();
		dist = in.readLong();
		distances.put(root, dist);
	    }

            int neighlen = in.readInt();
            neighbors = new ArrayList<Long>();            
            for (int j = 0; j < neighlen; j++) {
                neighbors.add(in.readLong());
            }
        }

        public String toString() {
            return neighbors.toString();
        }

    }

    /* The first mapper. Part of the graph loading process. */
    public static class LoaderMap extends Mapper<LongWritable, LongWritable, 
					  LongWritable, LongWritable> {

        @Override
	    public void map(LongWritable key, LongWritable value, Context context)
	    throws IOException, InterruptedException {
            context.write(key, value);
	    context.write(value, new LongWritable(-1L));
        }
    }


    /* The first reducer. */
    public static class LoaderReduce extends Reducer<LongWritable, LongWritable, 
					     LongWritable, Node> {

        public void reduce(LongWritable key, Iterable<LongWritable> values, 
			   Context context) throws IOException, InterruptedException {
            long denom = Long.parseLong(context.getConfiguration().get("denom"));
            Node n = new Node(key.get());
	    int count = 0;
            for (LongWritable value : values) {
		long val = value.get();
		if (val != -1) {
		    n.addNeighbor(val);
		    count += 1;
		}
            }
            double prob = 1.0 / denom;
            Random r = new Random();
            if (r.nextDouble() < prob) {
		if (count > 0) {
		    n.setStart();
		}
		n.setDistance(key.get(), 0);
            }
            context.write(key, n);
        }
    }


    public static class BFSMap extends Mapper<LongWritable, Node,
				       LongWritable, Node> {

	private LongWritable id = new LongWritable();

        @Override
	public void map(LongWritable key, Node value, Context context)
	    throws IOException, InterruptedException {
            if (value.isStart()) {
		HashMap<Long, Long> distances = value.getDistances();
		ArrayList<Long> roots = new ArrayList<Long>(distances.keySet());
                for (long neighbor : value.getNeighbors()) {
		    id.set(neighbor);
                    Node temp = new Node(neighbor);
		    if (roots.size() > 0) {
			for (long root : roots) {
			    temp.setDistance(root, distances.get(root) + 1);
			}
		    }
                    temp.setStart();
                    context.write(id, temp);
                }
            }
            context.write(key, value);
        }
    }

    public static class BFSReduce extends Reducer<LongWritable, Node, 
					  LongWritable, Node> {

        public void reduce(LongWritable key, Iterable<Node> values, 
	    Context context) throws IOException, InterruptedException {
            long sum = 0;
            Node output = new Node(key.get());
            for (Node node : values) {
		HashMap<Long, Long> distances = node.getDistances();
		ArrayList<Long> roots = new ArrayList<Long>(distances.keySet());
                if (node.getNeighbors().size() > 0) {
                    for (long neighbor : node.getNeighbors()) {
                        output.addNeighbor(neighbor);
                    }
                }
		if (roots.size() > 0) {
		    for (long root : roots) {
			if (node.getDistance(root) < output.getDistance(root)) {
			    output.setDistance(root, node.getDistance(root));
			}
		    }
		}
                if (node.isStart() && !output.isStart()) {
                    output.setStart();
                }
            }
	    context.write(key, output);
        }

    }

    public static class HistMap extends Mapper<LongWritable, Node,
					LongWritable, LongWritable> {
        private final static LongWritable ONE = new LongWritable(1L);
	private LongWritable dist = new LongWritable();

	public void map(LongWritable key, Node value, Context context)
	    throws IOException, InterruptedException {
	    ArrayList<Long> roots = new ArrayList<Long>(value.getDistances().keySet());
	    if (roots.size() > 0) {
		for (long root : roots) {
		    dist.set(value.getDistance(root));
		    context.write(dist, ONE);
		}
	    }
	}
    }

    public static class HistReduce extends Reducer<LongWritable, LongWritable,
					LongWritable, LongWritable> {
	public void reduce(LongWritable key, Iterable<LongWritable> values,
	    Context context) throws IOException, InterruptedException {
	    long sum = 0L;
            for (LongWritable value : values) {
                sum += value.get();
            }
            context.write(key, new LongWritable(sum));
	}
    }



    public static void main(String[] rawArgs) throws Exception {
        GenericOptionsParser parser = new GenericOptionsParser(rawArgs);
        Configuration conf = parser.getConfiguration();
        String[] args = parser.getRemainingArgs();

        // Pass in denom command line arg:
        conf.set("denom", args[2]);

        // Sample of passing value from main into Mappers/Reducers using
        // conf. You might want to use something like this in the BFS phase:
        // See LoaderMap for an example of how to access this value
        conf.set("inputValue", (new Integer(5)).toString());

        // Setting up mapreduce job to load in graph
        Job job = new Job(conf, "load graph");

        job.setJarByClass(SmallWorld.class);

        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(LongWritable.class);
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(Node.class);

        job.setMapperClass(LoaderMap.class);
        job.setReducerClass(LoaderReduce.class);

        job.setInputFormatClass(SequenceFileInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        // Input from command-line argument, output to predictable place
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path("bfs-0-out"));

        // Actually starts job, and waits for it to finish
        job.waitForCompletion(true);

        // Repeats your BFS mapreduce
        int i = 0;
        while (i < MAX_ITERATIONS) {
            job = new Job(conf, "bfs" + i);
            job.setJarByClass(SmallWorld.class);

            // Feel free to modify these four lines as necessary:
            job.setMapOutputKeyClass(LongWritable.class);
            job.setMapOutputValueClass(Node.class);
            job.setOutputKeyClass(LongWritable.class);
            job.setOutputValueClass(Node.class);

            // You'll want to modify the following based on what you call
            // your mapper and reducer classes for the BFS phase.
            job.setMapperClass(BFSMap.class); // currently the default Mapper
            job.setReducerClass(BFSReduce.class); // currently the default Reducer

            job.setInputFormatClass(SequenceFileInputFormat.class);
            job.setOutputFormatClass(SequenceFileOutputFormat.class);

            // Notice how each mapreduce job gets gets its own output dir
            FileInputFormat.addInputPath(job, new Path("bfs-" + i + "-out"));
            FileOutputFormat.setOutputPath(job, new Path("bfs-"+ (i+1) +"-out"));

            job.waitForCompletion(true);
            i++;
        }

        // Mapreduce config for histogram computation
        job = new Job(conf, "hist");
        job.setJarByClass(SmallWorld.class);

        // Feel free to modify these two lines as necessary:
        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(LongWritable.class);

        // DO NOT MODIFY THE FOLLOWING TWO LINES OF CODE:
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(LongWritable.class);

        // You'll want to modify the following based on what you call your
        // mapper and reducer classes for the Histogram Phase
        job.setMapperClass(HistMap.class); // currently the default Mapper
        job.setReducerClass(HistReduce.class); // currently the default Reducer

        job.setInputFormatClass(SequenceFileInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        // By declaring i above outside of loop conditions, can use it
        // here to get last bfs output to be input to histogram
        FileInputFormat.addInputPath(job, new Path("bfs-"+ i +"-out"));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.waitForCompletion(true);
    }
}
