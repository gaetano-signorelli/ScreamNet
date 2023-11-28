package com.example.screamify;

import android.app.Dialog;
import android.content.DialogInterface;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.fragment.app.DialogFragment;

public class SaveDialogFragment extends DialogFragment {

    private final float[] wave;
    private final MainActivity activity;

    private WavSaver saver;

    private ProgressDialogFragment progress_fragment;

    private String name;

    public SaveDialogFragment(float[] wave, MainActivity activity, ProgressDialogFragment progress_fragment) {
        super();

        this.wave = wave;
        this.activity = activity;
        this.progress_fragment = progress_fragment;

        name = null;
    }

    @Override
    public Dialog onCreateDialog(Bundle savedInstanceState) {

        super.onCreateDialog(savedInstanceState);

        LayoutInflater layout = LayoutInflater.from(getActivity());
        final View view = layout.inflate(R.layout.save_dialog, null);

        // Use the Builder class for convenient dialog construction
        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        builder.setMessage(R.string.save_dialog_title)
                .setPositiveButton(R.string.save_dialog_yes, new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        EditText name_box = view.findViewById(R.id.name_text);
                        name = name_box.getText().toString();
                        dismiss();
                    }
                })
                .setNegativeButton(R.string.save_dialog_no, new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        // User cancelled the dialog
                        dismiss();
                    }
                })
                .setView(view);
        // Create the AlertDialog object and return it
        return builder.create();
    }

    @Override
    public void onDismiss (DialogInterface dialog){
        if (name != null) save(name);
        else progress_fragment.dismiss();
    }

    private void save(String name){

        saver = new WavSaver(wave, name + ".wav", activity, progress_fragment);
        saver.start();
        try {
            saver.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        Toast.makeText(getActivity(), "Recording saved successfully in 'Recordings'", Toast.LENGTH_SHORT).show();
    }
}
