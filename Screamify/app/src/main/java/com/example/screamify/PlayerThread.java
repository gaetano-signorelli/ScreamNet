package com.example.screamify;

import android.media.AudioTrack;

import java.util.LinkedList;
import java.util.Queue;

public class PlayerThread extends Thread{

    private static final int WRITE_MODE = AudioTrack.WRITE_BLOCKING;

    private Queue<float[]> player_queue;
    private final AudioTrack player;
    private final MainActivity activity;
    private final boolean realtime;
    private final int tensor_size;

    public PlayerThread(AudioTrack player, MainActivity activity, boolean realtime, int tensor_size){
        this.player = player;
        this.activity = activity;
        this.realtime = realtime;
        this.tensor_size = tensor_size;

        player_queue = new LinkedList<>();
    }

    public void run(){

        startRotateButton();

        float[] wave = null;
        int count = 0;

        if (!realtime){
            wave = new float[tensor_size * (player_queue.size()+1)];
        }

        while(activity.isRecording() || !player_queue.isEmpty()){
            if (!player_queue.isEmpty()){
                float[] output = player_queue.poll();
                player.write(output,0,output.length, WRITE_MODE);
                if (!realtime){
                    System.arraycopy(output, 0, wave, (count*tensor_size), tensor_size);
                    count++;
                }
            }
        }
        stopRotateButton();
        player.stop();

        askToSave(wave);
    }

    public Queue<float[]> getQueue(){
        return player_queue;
    }

    private void startRotateButton(){
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                activity.startRotateButton();
            }
        });
    }

    private void stopRotateButton(){
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                activity.stopRotateButton();
            }
        });
    }

    private void askToSave(float[] waveform){
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                activity.askToSave(waveform);
            }
        });
    }
}
