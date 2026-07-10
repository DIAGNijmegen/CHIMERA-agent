#!/usr/bin/env bash
# ============================================================================
# Package the evaluator for upload to Grand Challenge.
#
#   STEP 1  (Re)build the image
#   STEP 2  docker save + gzip the image  -> chimera-evaluator_<timestamp>.tar.gz
#   STEP 3  tar the ground truth          -> ground_truth.tar.gz
#
# On Grand Challenge, upload the image tarball as the Evaluation Method and the
# ground_truth.tar.gz separately under Phase settings > Ground Truths. It is
# extracted to /opt/ml/input/data/ground_truth/ at runtime.
# ============================================================================

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-chimera-evaluator:latest}"

echo ""
echo "= STEP 1 = (Re)build the image"
export DOCKER_QUIET_BUILD=1
source "${SCRIPT_DIR}/do_build.sh"
echo "==== Done"
echo ""

# Derive a filename-safe timestamp from the image build time.
build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")
if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for image $DOCKER_IMAGE_TAG"
    exit 1
fi
formatted_build_info=$(echo "$build_timestamp" | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')

image_basename="${DOCKER_IMAGE_TAG%%:*}"
output_filename="${image_basename}_${formatted_build_info}.tar.gz"
output_path="${SCRIPT_DIR}/$output_filename"

echo "= STEP 2 = Saving the image"
echo "This can take a while."
docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$output_path"
printf "Saved as: \e[32m%s\e[0m\n" "${output_filename}"
echo "==== Done"
echo ""

echo "= STEP 3 = Packing the ground truth"
echo "This can take a while."
output_tarball_name="${SCRIPT_DIR}/ground_truth.tar.gz"
tar -czf "$output_tarball_name" -C "${SCRIPT_DIR}/ground_truth" .
printf "Saved as: \e[32mground_truth.tar.gz\e[0m\n"
echo "==== Done"
echo ""

printf "\e[31mIMPORTANT: upload ground_truth.tar.gz as a separate Ground Truth to your Phase!\e[0m\n"
