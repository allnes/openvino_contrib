# template-plugin

ArmGpu Plugin for Inference Engine which demonstrates basics of how Inference Engine plugin can be built and implemented on top of Inference Engine Developer Package and Plugin API.
As a backend for actual computations ngraph reference implementations is used, so the ArmGpu plugin is fully functional.

## How to build

```bash
$ cd $DLDT_HOME
$ mkdir $DLDT_HOME/build
$ cd $DLDT_HOME/build
$ cmake -DENABLE_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON ..
$ make -j8
$ cd $ARM_GPU_PLUGIN_HOME
$ mkdir $ARM_GPU_PLUGIN_HOME/build
$ cd $ARM_GPU_PLUGIN_HOME/build
$ cmake -DInferenceEngineDeveloperPackage_DIR=$DLDT_HOME/build ..
$ make -j8
```
