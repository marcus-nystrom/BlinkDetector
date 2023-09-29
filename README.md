# BlinkDetector
Code and data accompanying the paper Nyström, M., Andersson, R., Niehorster, D. C.,· Hessels, R. S., Hooge, I. T. C., (2023), What is a blink? Classifying and characterizing blinks in eye openness signals (submitted)

To test the blink algorithm, open and run the file `run_classification.py`.

Tested with Python 3.10.10, using Pandas (v. 1.5.3), Scipy (v. 1.10.0).

The `data`-folder contains two sub-folders: `spectrum` and `fusion`. The former contains data from the article and are recorded with the Tobii Pro Spectum at 600 Hz (using firmware 2.6.1). The latter folder contains sample data recorded with the Tobii Pro Fusion at 120 Hz (Fusion driver 2.5.4.0, Fusion firmware 96e7b52964, Set up using Eye Tracker Manager 2.6.0, Collected data on Pro Lab 1.217).
The fusion data are provided to examplify the performance of the blink detection algorihtm on data with lower sampling frequency.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
