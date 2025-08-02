echo "Running binary classification AutoGrading script..."
score=$(cat 6_deep_learning/CNN/tf_best_practices/binary_classification/marks.txt)
echo "score=$score" >> $GITHUB_OUTPUT