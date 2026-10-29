package dev.cometlab.simulator;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

public class MainActivity extends Activity {
    private WebView simulator;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().setStatusBarColor(Color.rgb(8,17,31));
        getWindow().setNavigationBarColor(Color.rgb(8,17,31));
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        simulator = new WebView(this);
        simulator.setBackgroundColor(Color.rgb(8,17,31));
        simulator.setOverScrollMode(View.OVER_SCROLL_NEVER);
        simulator.setWebViewClient(new WebViewClient());
        simulator.setWebChromeClient(new WebChromeClient());
        WebSettings settings = simulator.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setBuiltInZoomControls(false);
        settings.setDisplayZoomControls(false);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(false);
        setContentView(simulator);
        simulator.loadUrl("file:///android_asset/index.html");
    }

    @Override
    protected void onDestroy() {
        if (simulator != null) { simulator.destroy(); simulator = null; }
        super.onDestroy();
    }
}
