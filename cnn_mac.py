import numpy as np  # Importing NumPy for numerical operations

# ---------------------------
# Utility Functions
# ---------------------------

# Softmax normalization to convert raw scores into probabilities
def normalize_scores(raw_scores):
    adjusted = raw_scores - np.max(raw_scores)  # Numerical stability trick
    result = np.exp(adjusted)                   # Exponentiate scores
    return result / np.sum(result, axis=1, keepdims=True)  # Normalize to sum to 1

# Calculates the average cross-entropy loss (error)
def calculate_prediction_error(predictions, actual_values):
    batch_size = actual_values.shape[0]
    return -np.sum(np.log(predictions[np.arange(batch_size), actual_values])) / batch_size

# Checks if the model predicted the correct label
def check_prediction_success(predictions, actual_values):
    best_guesses = np.argmax(predictions, axis=1)
    return np.mean(best_guesses == actual_values)  # Returns accuracy as float


# ---------------------------
# Convolutional Layer (Conv 3x3)
# ---------------------------

class ImageFilter:
    def __init__(self, filter_count):
        self.num_patterns = filter_count  # Number of filters to use
        self.patterns = np.random.randn(filter_count, 3, 3) / 9  # Initialize random 3x3 filters

    def _scan_image_regions(self, picture):
        height, width = picture.shape
        for row in range(height - 2):  # Slide 3x3 window over image rows
            for col in range(width - 2):  # Slide over columns
                yield picture[row:row+3, col:col+3], row, col  # Yield patch and location

    def process_image(self, picture):
        self.original = picture  # Save original for backprop
        height, width = picture.shape
        result = np.zeros((height - 2, width - 2, self.num_patterns))  # Output volume

        for img_patch, row, col in self._scan_image_regions(picture):
            # Apply each filter to this patch
            result[row, col] = np.sum(img_patch * self.patterns, axis=(1, 2))  # Apply all filters
        return result  # Return feature map

    def update_patterns(self, error_gradient, learning_speed):
        pattern_updates = np.zeros(self.patterns.shape)  # Initialize pattern gradients

        for img_patch, row, col in self._scan_image_regions(self.original):
            for pattern_idx in range(self.num_patterns):  # For each filter
                # Accumulate gradients from backprop
                pattern_updates[pattern_idx] += error_gradient[row, col, pattern_idx] * img_patch

        self.patterns -= learning_speed * pattern_updates  # Update filter weights


# ---------------------------
# Pooling Layer (MaxPool 2x2)
# ---------------------------

class FeatureReducer:
    def _scan_features(self, features):
        height, width, depth = features.shape
        for row in range(0, height, 2):  # Step by 2 rows
            for col in range(0, width, 2):  # Step by 2 cols
                yield features[row:row+2, col:col+2], row//2, col//2  # Return 2x2 region

    def reduce_features(self, features):
        self.input_features = features  # Store input for backprop
        height, width, depth = features.shape
        reduced = np.zeros((height//2, width//2, depth))  # Pooled output

        for region, row, col in self._scan_features(features):
            reduced[row, col] = np.max(region, axis=(0, 1))  # Take max for each filter
        return reduced

    def backpropagate_error(self, error_gradient):
        input_gradient = np.zeros_like(self.input_features)  # Initialize backprop matrix

        for region, row, col in self._scan_features(self.input_features):
            height, width, filters = region.shape
            max_values = np.max(region, axis=(0, 1))  # Max in 2x2 region

            for i in range(height):
                for j in range(width):
                    for f in range(filters):
                        # Only pass error to the max pixel
                        if region[i, j, f] == max_values[f]:
                            input_gradient[row*2+i, col*2+j, f] = error_gradient[row, col, f]
        return input_gradient  # Return gradient to previous layer


# ---------------------------
# Output Layer (Fully Connected + Softmax)
# ---------------------------

class PredictionLayer:
    def __init__(self, input_size, output_classes):
        self.synapses = np.random.randn(input_size, output_classes) / input_size  # Initialize weights
        self.offsets = np.zeros(output_classes)  # Bias terms

    def predict(self, features):
        self.input_shape = features.shape
        self.flattened_input = features.flatten()  # Flatten 3D to 1D
        self.raw_output = np.dot(self.flattened_input, self.synapses) + self.offsets  # Linear transform
        return normalize_scores(self.raw_output.reshape(1, -1))  # Softmax to get probabilities

    def adjust_weights(self, output_gradient, learning_speed):
        for idx, grad in enumerate(output_gradient):
            if grad == 0:
                continue  # No need to update if zero gradient

            scores = np.exp(self.raw_output)
            total = np.sum(scores)

            # Derivative of softmax
            output_adjustment = -scores[idx] * scores / (total ** 2)
            output_adjustment[idx] = scores[idx] * (total - scores[idx]) / (total ** 2)

            weight_gradient = grad * output_adjustment
            self.synapses -= learning_speed * np.outer(self.flattened_input, weight_gradient)
            self.offsets -= learning_speed * weight_gradient

            # Return gradient to pass to the pooling layer
            return np.dot(self.synapses, weight_gradient).reshape(self.input_shape)


# ---------------------------
# CNNmac Model: Conv -> Pool -> Fully Connected -> Softmax
# ---------------------------

class CNNmac:
    def __init__(self):
        self.image_processor = ImageFilter(8)  # Use 8 filters
        self.feature_reducer = FeatureReducer()  # Max pooling
        self.classifier = PredictionLayer(13 * 13 * 8, 10)  # Fully connected layer to 10 classes

    def save(self, filename="learned_patterns.npz"):
        np.savez(filename, 
                 image_patterns=self.image_processor.patterns,
                 classifier_weights=self.classifier.synapses,
                 classifier_offsets=self.classifier.offsets)

    def load(self, filename="learned_patterns.npz"):
        saved_data = np.load(filename)
        self.image_processor.patterns = saved_data["image_patterns"]
        self.classifier.synapses = saved_data["classifier_weights"]
        self.classifier.offsets = saved_data["classifier_offsets"]

    def forward(self, image):
        # Run image through all layers to make a prediction
        features = self.image_processor.process_image(image[:, :, 0])  # Strip extra channel
        features = np.maximum(features, 0)  # Apply ReLU activation
        pooled = self.feature_reducer.reduce_features(features)  # Downsample features
        return self.classifier.predict(pooled)  # Classify pooled features

    def train(self, image_set, labels, epochs=5, learn_rate=0.005):
        for round_num in range(epochs):
            print(f"Training round {round_num+1}")
            total_error = 0
            success_count = 0

            for idx in range(len(image_set)):
                image = image_set[idx][:, :, 0]  # Remove extra channel
                features = self.image_processor.process_image(image)
                features = np.maximum(features, 0)
                pooled = self.feature_reducer.reduce_features(features)
                predictions = self.classifier.predict(pooled)

                # Track error and accuracy
                error = calculate_prediction_error(predictions, np.array([labels[idx]]))
                success = check_prediction_success(predictions, np.array([labels[idx]]))
                total_error += error
                success_count += success

                # Create gradient for target class
                correction = np.zeros(10)
                correction[labels[idx]] = -1 / predictions[0][labels[idx]]

                # Backpropagate error
                gradient = self.classifier.adjust_weights(correction, learn_rate)
                gradient = self.feature_reducer.backpropagate_error(gradient)
                self.image_processor.update_patterns(gradient, learn_rate)

                if idx % 100 == 99:
                    print(f"Progress: {idx+1} images | Avg error: {total_error/100:.3f} | Accuracy: {success_count:.2f}")
                    total_error = 0
                    success_count = 0
