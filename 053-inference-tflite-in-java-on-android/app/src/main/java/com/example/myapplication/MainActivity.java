package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.app.Activity;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.MappedByteBuffer;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MyActivity";

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;

    /** Optional NNAPI delegate for accleration. */
    private NnApiDelegate nnApiDelegate = null;

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.e(TAG, "Tensorflow Lite Start");
        //Activity activity = (Activity) getApplicationContext();
        Activity activity = MainActivity.this;

        nnApiDelegate = new NnApiDelegate();
        tfliteOptions.addDelegate(nnApiDelegate);

        try {
            tfliteModel = FileUtil.loadMappedFile(activity, "my_model.tflite");
        } catch (IOException e) {
            e.printStackTrace();
        }

        tflite = new Interpreter(tfliteModel, tfliteOptions);

        float[][] inputs = new float[1][1];
        inputs[0][0] = 15.0f;
        float[][] outputs = new float[1][1];
        tflite.run(inputs, outputs);
        Log.e(TAG, "Tensorflow Lite result : " + outputs[0][0]);

    }
}
