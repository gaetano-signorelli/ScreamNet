package com.example.screamify;

import android.os.Environment;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

public class WavSaver extends Thread{

    private static final int CHANNELS = 1;
    private static final int SAMPLE_RATE = 22050;
    private static final int BITS_PER_SAMPLE = 32;

    private final float[] wave;
    private final String name;
    private final MainActivity activity;

    private ProgressDialogFragment progress_fragment;
    private float progress;
    private float step_progress;

    public WavSaver(float[] wave, String name, MainActivity activity,ProgressDialogFragment progress_fragment){
        this.wave = wave;
        this.name = name;
        this.activity = activity;
        this.progress_fragment = progress_fragment;

        progress = 0.0f;
        step_progress = 100.0f / wave.length;
    }

    public void run(){

        File dir = new File(Environment.getExternalStorageDirectory(), "Recordings");
        if (!dir.exists()) {
            dir.mkdir();
        }

        File output = new File(dir, name);

        try {
            save(output, CHANNELS, SAMPLE_RATE, BITS_PER_SAMPLE);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private float[] remove_trailing_zeros(float[] wave){

        int n_trailing_zeros = 0;

        for(int i=wave.length-1; i>=0; i--){
            if (wave[i]==0.0) n_trailing_zeros++;
            else break;
        }

        int new_len = wave.length - n_trailing_zeros;
        float[] new_wave = new float[new_len];

        System.arraycopy(wave, 0, new_wave, 0, new_len);

        return new_wave;
    }

    private void save(File output, int channelCount, int sampleRate, int bitsPerSample) throws IOException {

        final int inputSize = (int) wave.length;

        try {
            OutputStream encoded = new FileOutputStream(output);
            // WAVE RIFF header
            writeToOutput(encoded, "RIFF"); // chunk id
            writeToOutput(encoded, 36 + inputSize*4); // chunk size
            writeToOutput(encoded, "WAVE"); // format

            // SUB CHUNK 1 (FORMAT)
            writeToOutput(encoded, "fmt "); // subchunk 1 id
            writeToOutput(encoded, 16); // subchunk 1 size
            writeToOutput(encoded, (short) 3); // audio format (3 = Float-32)
            writeToOutput(encoded, (short) channelCount); // number of channelCount
            writeToOutput(encoded, sampleRate); // sample rate
            writeToOutput(encoded, sampleRate * channelCount * bitsPerSample / 8); // byte rate
            writeToOutput(encoded, (short) (channelCount * bitsPerSample / 8)); // block align
            writeToOutput(encoded, (short) bitsPerSample); // bits per sample

            // SUB CHUNK 2 (AUDIO DATA)
            writeToOutput(encoded, "data"); // subchunk 2 id
            writeToOutput(encoded, inputSize*4); // subchunk 2 size

            putData(encoded);

            encoded.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

    private void writeToOutput(OutputStream output, String data) throws IOException {
        byte[] bytes = data.getBytes(StandardCharsets.US_ASCII);
        output.write(bytes);
    }

    private void writeToOutput(OutputStream output, int data) throws IOException {
        byte[] bytes = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(data).array();
        output.write(bytes);
    }

    private void writeToOutput(OutputStream output, short data) throws IOException {
        byte[] bytes = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(data).array();
        output.write(bytes);
    }

    private void putData(OutputStream output) throws IOException {
        for (float value : wave) {
            int int_representation = Float.floatToRawIntBits(value);
            writeToOutput(output, int_representation);

            int last_progress_approx = (int) progress;
            progress += step_progress;
            int progress_approx = (int) progress;
            if (progress_approx != last_progress_approx){
                activity.runOnUiThread(() -> {
                    progress_fragment.update(progress_approx);
                });
            }

        }
    }
}
