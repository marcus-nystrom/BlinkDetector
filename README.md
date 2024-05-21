# BlinkDetector
Code and data accompanying the paper Nyström, M., Andersson, R., Niehorster, D. C., Hessels, R. S., & Hooge, I. T. (2024). What is a blink? Classifying and characterizing blinks in eye openness signals. Behavior Research Methods, 1-20.

To test the blink algorithm, open and run the file `run_classification.py`.

Tested with Python 3.10.10, using Pandas (v. 1.5.3), Scipy (v. 1.10.0).

The version of the algorithm used in the papers is the following:
https://github.com/marcus-nystrom/BlinkDetector/tree/a2d8caea6c149555a8154d298ff6d686b750df75

The `data`-folder contains two sub-folders: `spectrum` and `fusion`. The `spectrum` folder contains data from the article and are recorded with the Tobii Pro Spectum at 600 Hz (using firmware 2.6.1). The `fusion` folder contains sample data recorded with the Tobii Pro Fusion at 120 Hz (Fusion driver 2.5.4.0, Fusion firmware 96e7b52964, set up using Eye Tracker Manager 2.6.0, data collected using Pro Lab 1.217).
The fusion data are provided to exemplify the performance of the blink detection algorithm on data with lower sampling frequency.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
