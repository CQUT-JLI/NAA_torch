# NAA PyTorch Implementation Notes

This document records the implementation details of the PyTorch reproduction of NAA converted from the original TensorFlow implementation.

The content can be used directly as README material or as an implementation note inside the repository.

---

## 1. Data Preprocessing

For the NIPS 2017 dataset, each input image is first resized according to the surrogate model input size. For Inception-style surrogate models, the image size is set to `299 × 299`.

In this PyTorch implementation, the dataset preprocessing is responsible for:

```python
image = Image.open(filepath).convert('RGB')
image = image.resize((image_size, image_size), Image.BILINEAR)
image = np.array(image).astype(np.float32) / 255.0
image = torch.from_numpy(image).permute(2, 0, 1)
```

Therefore, the image returned by `AdvDataset` has shape:

```text
[B, 3, 299, 299]
```

and its value range is:

```text
[0, 1]
```

The adversarial perturbation `delta` is also added in this `[0,1]` image space:

```python
adv_images = images + delta
```

For PyTorch and timm ImageNet models, class indices are `0–999`, while labels in the original NIPS dataset follow the `1–1000` convention. Therefore, all labels are shifted by `-1` during loading:

```python
label = TrueLabel - 1
```

For targeted attacks, both the true label and target label are shifted:

```python
true_label = TrueLabel - 1
target_label = TargetClass - 1
```

---

## 2. Model-Side Normalization

The model is wrapped as:

```python
model = nn.Sequential(preprocess, classifier)
```

where `PreprocessingModel` performs resizing and normalization before the actual classifier forward pass:

```python
x = normalize(resize(x))
```

For TensorFlow-style Inception models, the normalization is:

```python
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]
```

Thus, the input passed to the actual Inception model is:

```text
x_model = (x - 0.5) / 0.5 = 2x - 1
```

Although the PyTorch implementation adds perturbations in `[0,1]` space, the actual model input is still in `[-1,1]`, which is consistent with the original TensorFlow Inception preprocessing.

The original TensorFlow implementation first normalizes images into the model input space and then updates adversarial images in the normalized space. In contrast, this PyTorch implementation maintains the perturbation in `[0,1]` image space and normalizes `images + delta` before model forwarding.

For Inception models, the normalization is a linear transformation, so the two implementations are numerically equivalent after the corresponding epsilon and step-size scaling:

```text
Norm(x + delta) = 2(x + delta) - 1 = 2x - 1 + 2delta
```

Therefore, using `epsilon = 16/255` in `[0,1]` space is equivalent to using `epsilon = 2 * 16/255` in the TensorFlow normalized `[-1,1]` space.

---

## 3. Supported Surrogate Models

This implementation mainly uses TensorFlow-style pretrained models from `timm`:

```text
inception_v3.tf_in1k
inception_v4.tf_in1k
inception_resnet_v2.tf_in1k
```

The corresponding model names in this repository are:

```text
tf_inception_v3
tf_inception_v4
tf_inception_resnet_v2
```

For these models, the input size is `299 × 299`, and the preprocessing uses:

```python
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]
```

which matches the TensorFlow-style Inception input convention.

---

## 4. Feature Layer Mapping

The original TensorFlow NAA implementation obtains intermediate feature tensors by searching graph operation names. In PyTorch, intermediate features are obtained using forward hooks.

The following TensorFlow layer names are mapped to PyTorch/timm module names:

```python
{
    "InceptionV3/InceptionV3/Mixed_5b/concat": "Mixed_5b",
    "InceptionV4/InceptionV4/Mixed_5e/concat": "features.9",
    "InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu": "conv2d_4a"
}
```

During the forward pass, hooks are registered on the selected modules:

```python
handles = [mod.register_forward_hook(hook_fn) for mod in self.opt_operations]
```

Each time the model performs a forward pass, the output of the selected feature layer is stored and used to compute the NAA attribution loss.

---

## 5. Forward Pass and NAA Computation

The PyTorch implementation follows the same main logic as the original TensorFlow implementation:

```text
1. Compute baseline feature.
2. Estimate neuron attribution weights by aggregated gradients.
3. Construct NAA loss.
4. Compute gradients with respect to the adversarial perturbation.
5. Update perturbation with momentum.
6. Optionally apply DI and PI.
7. Clip perturbation into the epsilon-ball.
```

### 5.1 Baseline Feature

A black image is used as the baseline:

```python
baseline = torch.zeros_like(images)
```

The baseline is forwarded through the model, and the selected intermediate layer feature is stored:

```python
base_features = [f.clone().detach() for f in features]
```

This corresponds to the baseline feature used in the original NAA attribution formulation.

### 5.2 Neuron Attribution Weight Estimation

For each input image, NAA estimates the importance weight of neurons by computing the gradient of the original class probability with respect to the selected intermediate feature layer.

In the PyTorch implementation:

```python
probs = F.softmax(logits, dim=1)
target_response = probs.gather(1, labels.view(-1, 1)).squeeze(1)
grad_fs = torch.autograd.grad(target_response.sum(), features)
```

This corresponds to the TensorFlow implementation:

```python
weights_tensor = tf.gradients(tf.nn.softmax(logits) * label_ph, opt_operations[0])[0]
```

The gradients are accumulated over `ens` integration samples:

```python
agg_grads[k] += grad_f / self.ens
```

Then the accumulated gradients are normalized and negated:

```python
weights_list = [-normalize(ag.detach(), opt=2) for ag in agg_grads]
```

The negative sign follows the original NAA design: the attack aims to suppress or destroy the attribution contribution of neurons that support the original class.

One minor implementation difference is that the PyTorch version explicitly divides the accumulated gradient by `ens`, while the TensorFlow version directly accumulates all gradients and normalizes them afterward. Since the final weight is L2-normalized, this scaling does not affect the attack direction.

### 5.3 NAA Loss

The NAA loss is computed as:

```python
attribution = (adv_feat - base_feat) * weights
```

That is, the attribution of the adversarial sample is measured by the feature difference between the adversarial feature and the baseline feature, weighted by the pre-computed neuron attribution weights.

The loss is then used to update the perturbation:

```python
grad = torch.autograd.grad(loss, delta)[0]
```

Compared with the TensorFlow implementation, the PyTorch version directly optimizes `delta`, while the TensorFlow version directly updates the normalized adversarial image `images_adv`.

---

## 6. Difference from the TensorFlow Implementation

The PyTorch implementation is not a line-by-line translation of the TensorFlow static graph code. Instead, it keeps the same NAA computation while adapting the implementation to PyTorch's dynamic computation graph.

### 6.1 Batch Organization

The TensorFlow implementation uses placeholders:

```python
ori_input = tf.placeholder(...)
adv_input = tf.placeholder(...)
```

and concatenates clean and adversarial images:

```python
x = tf.concat([ori_input, adv_input], axis=0)
```

Thus, the TensorFlow graph forwards a `2B` batch.

The PyTorch implementation does not concatenate clean and adversarial images. Instead, it forwards baseline images, integration-path images, and adversarial images separately. This is more natural in PyTorch and avoids unnecessary graph construction.

### 6.2 Attack Variable

The TensorFlow implementation updates `images_adv` directly in the normalized model input space.

The PyTorch implementation maintains a perturbation variable:

```python
delta = torch.zeros_like(images, requires_grad=True)
```

and constructs adversarial images by:

```python
adv_images = images + delta
```

The perturbation is clipped by:

```python
delta = torch.clamp(delta, -epsilon, epsilon)
```

This is equivalent to the TensorFlow clipping operation:

```python
images_adv = clip(images_adv, images_tmp - eps, images_tmp + eps)
```

under the corresponding input-space scaling.

### 6.3 Weight Aggregation

The TensorFlow implementation accumulates `ens` gradients and normalizes the final result. The PyTorch implementation averages the gradients during accumulation:

```python
agg_grads[k] += grad_f / self.ens
```

Since the final attribution weight is L2-normalized, this does not change the attack direction.

### 6.4 DI Padding Value

The original TensorFlow DI implementation performs padding with value `0` in the normalized input space. For Inception models, normalized value `0` corresponds to gray value `0.5` in `[0,1]` image space.

Therefore, the PyTorch implementation pads with `0.5` in `[0,1]` space for TensorFlow-style Inception models:

```python
self.di_pad_value = preprocess.input_mean
```

For Inception models:

```text
pad value in PyTorch image space = 0.5
pad value after normalization     = 0
```

This matches the TensorFlow DI behavior.

### 6.5 PI / PIM

The PI update follows the original TensorFlow implementation. The main difference is tensor layout:

```text
TensorFlow: NHWC + depthwise_conv2d
PyTorch:    NCHW + grouped conv2d
```

The projection kernel and update rule are kept consistent with the original implementation.

---

## 7. Parameter Alignment

For TensorFlow-style Inception surrogate models, the main attack parameters are aligned with the TensorFlow implementation:

```text
max_epsilon          = 16
num_iter             = 10
alpha                = 1.6
momentum             = 1.0
image_size           = 299
image_resize         = 331
prob                 = 0.7
amplification_factor = 2.5
gamma                = 0.5
Pkern_size           = 3
ens                  = 30
```

In the PyTorch implementation, `epsilon` and `alpha` are defined in `[0,1]` image space:

```python
epsilon = max_epsilon / 255.0
alpha   = alpha / 255.0
```

For TensorFlow-style Inception models, the model input is obtained by:

```text
x -> 2x - 1
```

so these values are equivalent to the TensorFlow normalized-space settings:

```python
epsilon_tf = 2 * max_epsilon / 255.0
alpha_tf   = 2 * alpha / 255.0
```

Thus, although the perturbation is applied in different spaces, the effective perturbation budget seen by the Inception model is aligned with the TensorFlow implementation.

---

## 8. Summary

In summary, this PyTorch implementation preserves the core mechanism of the original NAA attack:

```text
neuron attribution weight estimation
+ attribution-guided feature disruption
+ momentum-based iterative update
+ optional DI and PI enhancement
```

The main implementation changes are introduced to match PyTorch's dynamic graph style and the timm model interface. For TensorFlow-style Inception surrogate models, the preprocessing, perturbation budget, feature-layer selection, DI behavior, PI update, and main attack hyperparameters are aligned with the original TensorFlow implementation.

The reproduced attack performance is comparable to or better than the original reported results.
