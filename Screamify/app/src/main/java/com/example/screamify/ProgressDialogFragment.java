package com.example.screamify;

import android.app.Dialog;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ProgressBar;

import androidx.appcompat.app.AlertDialog;
import androidx.fragment.app.DialogFragment;

public class ProgressDialogFragment extends DialogFragment {

    private ProgressBar bar;

    public ProgressDialogFragment() {
        super();
    }

    @Override
    public Dialog onCreateDialog(Bundle savedInstanceState) {

        super.onCreateDialog(savedInstanceState);

        LayoutInflater layout = LayoutInflater.from(getActivity());
        final View view = layout.inflate(R.layout.progress_dialog, null);
        bar = view.findViewById(R.id.progressBar);

        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        builder.setMessage(R.string.progress_dialog_title).setView(view);
        return builder.create();
    }

    public void update(int progress){
        bar.setProgress(progress);
    }
}
