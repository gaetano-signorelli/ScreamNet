package com.example.screamify;

import org.tensorflow.lite.Interpreter;

import java.util.Queue;

public class InferenceThread extends Thread{

    private final Interpreter interpreter;
    private final float[] input;
    private final int TENSOR_SIZE;
    private final Queue<float[]> player_queue;

    private final boolean REMOVE_SILENCE;
    private final int MIN_SILENCE_COUNT = 18000;
    private final float SILENCE_THRESHOLD = 0.0005f;

    public InferenceThread(Interpreter interpreter, float[] input, PlayerThread player_thread, boolean remove_silence){
        this.interpreter = interpreter;
        this.input = input;

        player_queue = player_thread.getQueue();

        TENSOR_SIZE = input.length;
        REMOVE_SILENCE = remove_silence;
    }

    public void run(){

        if (REMOVE_SILENCE){
            if (!isSilence(input)){
                inference();
            }
        }

        else{
            inference();
        }
    }

    private void inference(){

        float[] output_waveform = new float[TENSOR_SIZE];

        float[][] model_input = new float[1][TENSOR_SIZE];
        model_input[0] = input;

        float[][] model_output = new float[1][TENSOR_SIZE];
        model_output[0] = output_waveform;

        interpreter.run(model_input, model_output);

        float[] output = model_output[0];
        player_queue.add(output);
    }


    private boolean isSilence(float[] wave){
        int count = 0;

        for (int i=0; i<wave.length; i++){
            if (Math.abs(wave[i])<=SILENCE_THRESHOLD) count++;
        }

        return count>=MIN_SILENCE_COUNT;
    }
}
