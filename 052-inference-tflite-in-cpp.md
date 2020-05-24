# 1. How to build
bazel build
```
$ bazel build //tensorflow/lite/examples/minimal:minimal
```
binary output
```
tensorflow/bazel-bin/tensorflow/lite/examples/minimal/minimal
```

# 2. add input and output in minimal.cc
```
diff --git a/tensorflow/lite/examples/minimal/minimal.cc b/tensorflow/lite/examples/minimal/minimal.cc
index ba142da799..1019dc37ac 100644
--- a/tensorflow/lite/examples/minimal/minimal.cc
+++ b/tensorflow/lite/examples/minimal/minimal.cc
@@ -62,6 +62,7 @@ int main(int argc, char* argv[]) {

   // Fill input buffers
   // TODO(user): Insert code to fill input tensors
+  interpreter->typed_input_tensor<float>(0)[0] = 10;

   // Run inference
   TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
@@ -70,6 +71,8 @@ int main(int argc, char* argv[]) {

   // Read output buffers
   // TODO(user): Insert getting data out code.
+  float result = interpreter->typed_output_tensor<float>(0)[0];
+  fprintf(stderr, "minimal : %d\n", int(result));

   return 0;
 }
```
