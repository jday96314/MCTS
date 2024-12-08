package analyzeGameStartingPosition;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;
import java.io.FileInputStream;
import java.io.InputStream;

import game.Game;
import main.CommandLineArgParse;
import main.CommandLineArgParse.ArgOption;
import main.CommandLineArgParse.OptionTypes;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.trial.Trial;
import utils.AIFactory;
import other.move.Move;
import search.mcts.MCTS;

/**
 * Implementation of a utility to evaluate a game's starting position using a single agent.
 * The agent will be called to select an action and then estimate the game state value.
 * This result will be written to an output file.
 * 
 * Primarily meant for assessing if the game is fair by evaluating the initial game state.
 * 
 * Author: o1-mini, with small manual edits.
 */
public class AnalyzeGameStartingPosition
{
	//-------------------------------------------------------------------------
	
	/** Game name or file path */
	protected String gameName;

	/** Ruleset (if applicable) */
	protected String ruleset;

	/** The agent string (representing a single AI agent) */
	protected String agentString;

	/** Filepath to write evaluation results */
	protected String outputFilepath;

	/** Whether to treat the game name as a filepath */
	protected boolean treatGameNameAsFilepath;
	
	/** Amount of time to spend evaluating the position, in seconds. */
	protected double thinkingTime;

	//-------------------------------------------------------------------------
	
	/**
	 * Starts the evaluation experiment
	 */
	public void startEvaluation()
	{
		try
		{
			// Load the game
			final Game game;
			if (treatGameNameAsFilepath)
			{
				// Load the game from a file path
				if (ruleset != null && !ruleset.equals(""))
					game = GameLoader.loadGameFromFile(new File(gameName), ruleset);
				else
					game = GameLoader.loadGameFromFile(new File(gameName), new ArrayList<String>()); // No options
			}
			else
			{
				// Load the game by name
				if (ruleset != null && !ruleset.equals(""))
					game = GameLoader.loadGameFromName(gameName, ruleset);
				else
					game = GameLoader.loadGameFromName(gameName, new ArrayList<String>()); // No options
			}

			// Clear some memory
			game.description().setParseTree(null);
			game.description().setExpanded(null);
			
			final int numPlayers = game.players().count();

			if (numPlayers < 2)
			{
				System.err.println("Expected at least 2 players, but the game only has " + numPlayers + " players. Aborting.");
				return;
			}

			// Create the AI
			final MCTS ai = (MCTS) AIFactory.createAI(agentString);
			ai.setAutoPlaySeconds(this.thinkingTime);
			
			// Setup game context
			final Trial trial = new Trial(game);
			final Context context = new Context(game, trial);
			game.start(context);
			
			// Initialize AI
			ai.initAI(game, 1);

			// Have AI select a move
			final double maxSeconds = this.thinkingTime;
			final int maxIterations = -1; // -1 for unlimited iterations.
			final int maxDepth = -1; // -1 for unlimited depth.
			Move move = ai.selectAction(game, context, maxSeconds, maxIterations, maxDepth);
			
			// Get analysis results
			final double valueEstimate = ai.estimateValue();
			final int mctsIterationCount = ai.getNumMctsIterations();
			final int mctsPlayoutActionCount = ai.getNumPlayoutActions();
			
			/*
			// Write result to output file
			try (final PrintWriter writer = new PrintWriter(new FileWriter(outputFilepath, true))) {
				writer.println("Game: " + game.name());
				writer.println("Agent: " + ai.friendlyName());
				writer.println("Starting Position Evaluation: " + valueEstimate);
				writer.println("========================================");
			}
			*/
			File outputFile = new File(outputFilepath);

			// Ensure the directory exists
			File parentDir = outputFile.getParentFile();
			if (parentDir != null && !parentDir.exists()) {
			    parentDir.mkdirs(); // Create the directory if it doesn't exist
			}

			// Write result to output file
			try (final PrintWriter writer = new PrintWriter(new FileWriter(outputFile, true))) {
			    writer.println("Game: " + game.name());
			    writer.println("Agent: " + ai.friendlyName());
			    writer.println("Starting Position Evaluation: " + valueEstimate);
			    writer.println("MCTS playout action count: " + mctsPlayoutActionCount);
			    writer.println("MCTS iteration count: " + mctsIterationCount);
			    writer.println("========================================");
			}

			System.out.println("Evaluation written to: " + outputFilepath);
		}
		catch (final Exception e)
		{
			e.printStackTrace();
		}
	}
	
	//-------------------------------------------------------------------------
	
	/**
	 * Utility to load the configuration from a JSON file.
	 * 
	 * @param filepath Path to the JSON config file.
	 * @return The configuration object.
	 */
	public static AnalyzeGameStartingPosition fromJson(final String filepath)
	{
		try (final InputStream inputStream = new FileInputStream(new File(filepath)))
		{
			final JSONObject json = new JSONObject(new JSONTokener(inputStream));
			
			final String gameName = json.getString("gameName");
			final String ruleset = json.optString("ruleset", null);
			final String agentString = json.getString("agentString");
			final String outputFilepath = json.getString("outputFilepath");
			final double thinkingTime = json.getDouble("thinkingTime");
			final boolean treatGameNameAsFilepath = json.optBoolean("treatGameNameAsFilepath", false);
			
			final AnalyzeGameStartingPosition evaluator = new AnalyzeGameStartingPosition();
			evaluator.gameName = gameName;
			evaluator.ruleset = ruleset;
			evaluator.agentString = agentString;
			evaluator.outputFilepath = outputFilepath;
			evaluator.thinkingTime = thinkingTime;
			evaluator.treatGameNameAsFilepath = treatGameNameAsFilepath;
			
			return evaluator;
		}
		catch (final Exception e)
		{
			e.printStackTrace();
			return null;
		}
	}

	//-------------------------------------------------------------------------
	
	/**
	 * Main method for running the evaluator from command line.
	 * @param args Command line arguments.
	 */
	public static void main(final String[] args)
	{
		final CommandLineArgParse argParse = new CommandLineArgParse
		(
			true,
			"Single agent evaluates the starting position of various games and saves the starting position evaluations."
		);
		
		argParse.addOption(new ArgOption()
				.withNames("--json-files")
				.help("JSON configuration file describing the experiment.")
				.withNumVals("+")
				.withType(OptionTypes.String)
				.withDefault(Arrays.asList("/home/jday/Downloads/Ludii/test_config.json", "/home/jday/Downloads/Ludii/test_config_2.json")));
		
		// Parse the arguments
		if (!argParse.parseArguments(args))
			return;

		// Process configs.
		final List<String> jsonFilepaths = (List<String>) argParse.getValue("--json-files");
		
		for (String filepath : jsonFilepaths)
		{
			final AnalyzeGameStartingPosition evaluator = AnalyzeGameStartingPosition.fromJson(filepath);
			
			if (evaluator != null)
			{
				// Start the evaluation
				evaluator.startEvaluation();
			}
		}
	}
}