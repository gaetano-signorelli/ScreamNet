package com.example.screamify;

import android.media.AudioRecord;
import android.media.AudioTrack;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

public class RecordLoop extends Thread{

    private static final int READ_MODE = AudioRecord.READ_BLOCKING;
    private static final int TENSOR_SIZE = 22050;
    private static final boolean REMOVE_SILENCE = false;

    private final AudioRecord recorder;
    private final AudioTrack player;
    private final float[] recorder_buffer;
    private final int BUFFER_SIZE;
    private final boolean REALTIME;

    private final MainActivity activity;

    private final Interpreter interpreter;

    private PlayerThread player_thread;
    private InferenceThread inference_thread;

    public RecordLoop(AudioRecord recorder, AudioTrack player, float[] recorder_buffer,
                      int buffer_size, boolean realtime, MainActivity activity, Interpreter interpreter){
        this.recorder = recorder;
        this.player = player;
        this.recorder_buffer = recorder_buffer;
        this.activity = activity;
        this.interpreter = interpreter;

        BUFFER_SIZE = buffer_size;
        REALTIME = realtime;

        player_thread = new PlayerThread(player, activity, REALTIME, TENSOR_SIZE);
    }

    public void run(){

        if (REALTIME) player_thread.start();
        record_play_loop();

        if (REALTIME){
            try {
                player_thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        try {
            if (inference_thread!=null && inference_thread.isAlive()) inference_thread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void record_play_loop(){

        int float_read = recorder.read(recorder_buffer, 0, BUFFER_SIZE, READ_MODE);

        while(float_read>0){

            if (float_read>=TENSOR_SIZE && activity.isRecording()){
                //float[] input_waveform = new float[TENSOR_SIZE];
                //System.arraycopy(recorder_buffer, 0, input_waveform, 0, TENSOR_SIZE);
                float[] input_waveform = average(recorder_buffer, 3, 2, TENSOR_SIZE);
                inference_thread = new InferenceThread(interpreter, input_waveform, player_thread, REMOVE_SILENCE);
                inference_thread.start();
            }

            float_read = recorder.read(recorder_buffer, 0, BUFFER_SIZE, READ_MODE);
        }

        if (float_read<0) {
            Toast toast = Toast.makeText(activity, "An error occurred: " + float_read, Toast.LENGTH_SHORT);
            toast.show();
        }

        if (!REALTIME){
            player.play();
            player_thread.start();
        }
    }

    private float[] average(float[] signal, int window_size, int stride, int size){

        int start = (window_size-1) / 2;
        int end = Math.min(signal.length, size*stride) - start;
        float[] result = new float[size];

        int pos = 0;

        for(int i=0; i<start; i+=stride){
            result[pos] = signal[i];
            pos++;
        }

        for (int i=start; i<end; i+=stride){
            int a = i - start;
            int b = i + start;
            float sum = 0.0f;
            for (int j=a; j<=b; j++){
                sum += signal[j];
            }
            float avg = sum / window_size;
            result[pos] = avg;
            pos++;
        }

        for(int i=end; i<end+start && pos<size; i+=stride){
            result[pos] = signal[i];
            pos++;
        }

        return result;
    }
}
