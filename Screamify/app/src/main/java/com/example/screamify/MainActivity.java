package com.example.screamify;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioAttributes;
import android.media.AudioDeviceInfo;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.LinearInterpolator;
import android.view.animation.RotateAnimation;
import android.widget.ImageButton;
import android.widget.Switch;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private static final int SAMPLE_RATE = 22050;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_FLOAT;
    private static final int BUFFER_SIZE_RECORDER = 44100;
    private static final int BUFFER_SIZE_PLAYER = 22052;

    private static final int MICROPHONE_REQUEST_CODE = 10;

    private static final String MODEL_NAME = "screamer.tflite";

    private AudioRecord recorder;
    private AudioTrack player;
    private float[] recorder_buffer;
    private boolean recording;

    private RecordLoop record_thread;

    private ImageButton record_button;
    private Switch realtime_switch;

    private Interpreter interpreter;

    private boolean permission_granted;

    private RotateAnimation rotate_anim;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        permission_granted=false;

        initialize();

        record_button = findViewById(R.id.record_button);
        record_button.setImageResource(R.drawable.record_button_stop);

        realtime_switch = findViewById(R.id.realtime_switch);

        rotate_anim = new RotateAnimation(0f, 360f, Animation.RELATIVE_TO_SELF, 0.5f, Animation.RELATIVE_TO_SELF, 0.5f);
        rotate_anim.setDuration(3000);
        rotate_anim.setInterpolator(new LinearInterpolator());
        rotate_anim.setRepeatCount(Animation.INFINITE);

        record_button.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if (event.getAction() == MotionEvent.ACTION_DOWN) {
                    startRecording();
                } else if (event.getAction() == MotionEvent.ACTION_UP) {
                    try {
                        stopRecording();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                return true;
            }
        });
    }

    private void initialize(){
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[] {android.Manifest.permission.RECORD_AUDIO}, MICROPHONE_REQUEST_CODE);
        }
        else{
            permission_granted=true;
        }

        recording = false;

        recorder_buffer = new float[BUFFER_SIZE_RECORDER];

        recorder = new AudioRecord(
                MediaRecorder.AudioSource.CAMCORDER,
                SAMPLE_RATE*2,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                BUFFER_SIZE_RECORDER
        );
        AudioManager audioManager = (AudioManager) getBaseContext().getSystemService(Context.AUDIO_SERVICE);
        for (AudioDeviceInfo device : audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)) {
            if (device.getType() == AudioDeviceInfo.TYPE_BUILTIN_MIC) {
                recorder.setPreferredDevice(device);
                break;
            }
        }

        player = new AudioTrack.Builder()
                .setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .build())
                .setAudioFormat(new AudioFormat.Builder()
                        .setEncoding(AUDIO_FORMAT)
                        .setSampleRate(SAMPLE_RATE)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build())
                .setBufferSizeInBytes(BUFFER_SIZE_PLAYER)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build();

        try {
            MappedByteBuffer model_file = loadModelFile(getAssets(), MODEL_NAME);
            interpreter = new Interpreter(model_file);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void checkPermissions(){
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[] {android.Manifest.permission.RECORD_AUDIO}, MICROPHONE_REQUEST_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions,
                                           int[] grantResults)
    {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == MICROPHONE_REQUEST_CODE) {

            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                permission_granted = true;
                Toast.makeText(MainActivity.this, "Press \"Recording\" to start", Toast.LENGTH_SHORT).show();
            }
            else {
                permission_granted = false;
                Toast.makeText(MainActivity.this, "Permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void startRecording(){

        if (player.getPlayState()==AudioTrack.PLAYSTATE_PLAYING) return;

        boolean realtime = realtime_switch.isChecked();

        if (permission_granted){

            recording = true;

            record_button.setImageResource(R.drawable.record_button_start);
            recorder.startRecording();

            if (realtime){
                player.play();
            }

            record_thread = new RecordLoop(recorder, player, recorder_buffer, BUFFER_SIZE_RECORDER, realtime, this, interpreter);
            record_thread.start();
        }

        else{
            checkPermissions();
        }
    }

    private void stopRecording() throws InterruptedException {

        if (permission_granted){

            recording = false;

            recorder.stop();

            record_thread.join();

            recorder_buffer = new float[BUFFER_SIZE_RECORDER];
        }
    }

    private MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public boolean isRecording(){
        return recording;
    }

    public void startRotateButton(){
        record_button.startAnimation(rotate_anim);
    }

    public void stopRotateButton(){
        record_button.clearAnimation();
        record_button.setImageResource(R.drawable.record_button_stop);
    }

    public void askToSave(float[] waveform){

        ProgressDialogFragment progress_fragment = new ProgressDialogFragment();
        progress_fragment.show(getSupportFragmentManager(), "progress");
        SaveDialogFragment dialog = new SaveDialogFragment(waveform, this, progress_fragment);
        dialog.show(getSupportFragmentManager(), "save");
    }
}